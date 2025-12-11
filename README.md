# Assignment 3 – Report

## Part 1: Is an LLM smart enough to play “tic-tac-what?”

### 1. Problem Formulation

In Part 1, I study whether a large language model (Llama3 via Ollama) can play a misère variant of tic-tac-toe called **“tic-tac-what?”** (very similar to Notakto).

**Game rules (as used in my implementation):**

* The game is played on an (N \times N) grid (in this assignment mainly (3 \times 3); optionally extendable to (4 \times 4)).
* Both players use the same symbol `"x"`. There are **no** `"o"` markers.
* Empty cells are represented as `"."`.
* On each turn, a player **must** place exactly one `"x"` in an empty cell. Passing is not allowed.
* The game is misère: the player who **creates a line** of length (k) (horizontal, vertical, or diagonal) consisting only of `"x"` **immediately loses**.

  * In the provided tests, (N = 3) and (k = 3).

I treat the LLM as a **policy** that, given a board state, proposes a move. The main questions I try to answer are:

1. On the five given 3×3 test boards (from NOTE.md), can the LLM choose the “correct” moves (i.e., avoid immediate loss when a safe move exists)?
2. How do different prompt designs affect the LLM’s ability to follow the rules and pick good moves?
3. From an empty 3×3 board, can the LLM beat a random opponent, and how many moves can it play before it starts making obvious mistakes (e.g., illegal moves or immediate losing moves)?

---

### 2. Implementation

#### 2.1 Environment and model

* I use **Google Colab** with a GPU runtime as suggested.
* I run a local **Llama3** model via **Ollama** and connect to it from the notebook.
* I use **LangChain** and **Pydantic** for structured outputs:

  * `langchain_core.output_parsers.PydanticOutputParser`
  * `pydantic.BaseModel` to define schemas.

Some minor environment issues occurred (e.g., missing `langchain.prompts` in newer versions); I fixed these either by:

* switching imports to `langchain_core.prompts` when needed, or
* **removing the dependency on `ChatPromptTemplate` entirely** and building prompt strings manually.

#### 2.2 Board representation and helper functions

* Internally, a board is represented as a list of lists of strings `List[List[str]]`, where each element is `"x"` or `"."`.
* For convenience I define:

  * `strings_to_board(rows: List[str]) -> List[List[str]]` – convert from `["x..", ".x.", "xx."]` style to 2D list.
  * `board_to_strings(board: List[List[str]]) -> List[str]`.
  * `print_board(board)` – pretty-print the board.

To evaluate moves, I implement:

* `has_k_in_a_row(board, k)` – checks whether the board currently contains **any** line of length `k` of `"x"` (horizontal, vertical, or diagonal).
* `find_safe_moves(board, k)` – enumerates all legal moves (empty cells `"."`), temporarily places an `"x"` there, and returns those that **do not** create a losing line of length `k`.

This **solver** provides a ground truth for what a “safe” move is in each test case and is used to check whether the LLM’s move matches any safe move.

#### 2.3 Prompt design: three iterations

I tried three main prompt designs.

---

##### Prompt v1 – Full-board JSON output (baseline)

My initial attempt followed the starter code’s idea: the model had to output the **entire new board** as a JSON object:

```json
{
  "board": [
    ["x", ".", "."],
    [".", "x", "."],
    ["x", "x", "."]
  ]
}
```

A Pydantic schema enforced `board: List[List[str]]`, square dimensions (N \times N), and cells in `{"x", "."}`. I passed the automatically generated `parser.get_format_instructions()` into the prompt, which described the JSON schema in detail.

**Observed problems:**

* The model often produced boards like `{"board": ["x..", ".x.", "xx."]}` or nested lists with entries `"xx."`, `"x.x"`, etc. These violate the schema (each cell must be a single character).
* Pydantic failed to parse many outputs. I added a manual **fallback** function that tries to “salvage” something like `["x..", ".x.", "xx."]` into a (3\times3) board by splitting strings into characters and trimming/padding with `"."`, but this is already a sign that the model is not following the format reliably.
* Even after salvaging, the model often **changed multiple cells at once**, essentially “rewriting” the entire board instead of placing one `"x"`.

Result: under this prompt, all 5 test cases **failed** under strict evaluation.

---

##### Prompt v2 – Improved full-board prompt with explicit format rules

To address the format issues, I replaced the long, generic schema instructions with a **hand-crafted** description, plus a clear JSON example:

* Only one top-level key `"board"`.
* `"board"` must be an (N \times N) list of lists.
* Each element must be exactly `"x"` or `"."`.
* Strings like `"x.."` or `"xx."` are explicitly forbidden as cells.
* The model should “think step by step in your head” but only output the final JSON.

This significantly improved the **format correctness**:

* In 4 out of 5 test cases, the raw LLM output was now a valid JSON board that passed Pydantic without needing fallback.
* However, in all of these cases, the resulting board differed from the original in **2–5 cells**, not just one. The model was still “repainting” the board, e.g.:

  * Zeroing out several existing `"x"`;
  * Moving existing `"x"` around;
  * Generating symmetric-looking positions that obey some aesthetic property but violate the one-move rule.

Result: still **0/5** test cases passed under a strict definition (exactly one move, no immediate loss).

---

##### Prompt v3 – Coordinate-only output (`row`, `col`)

Finally, I changed the **structure** of the task itself: instead of asking the model to output the entire board, I only ask it to output the **coordinates of one move**.

I define a Pydantic model:

```python
class TicTacWhatMove(BaseModel):
    row: int
    col: int
    reason: Optional[str] = None
```

The prompt now says (paraphrased):

* You are an expert player in this misère tic-tac-toe variant.
* Game rules: same as before (both players use `"x"`, you lose if you create a line of length `k`).
* **Coordinate system**: 0-based indexing; `row=0` top row, `col=0` leftmost column; board size is (N).
* You **must** choose a cell `(row, col)` such that `board[row][col] == "."`.
* Output format: a single JSON object with `"row"` and `"col"` (and optional `"reason"`), e.g. `{ "row": 1, "col": 2 }`.

The model is instructed to:

1. Internally list all empty cells `"."`.
2. Internally simulate placing `"x"` in each empty cell and check whether that move creates a losing line of length `k`.
3. If there is at least one safe move, choose one safe move; otherwise choose any legal move.
4. Return **only** the JSON coordinates.

Then, in **Python code**, I apply the move:

* Check if `0 ≤ row, col < N` and if the cell is empty.
* Set `board[row][col] = "x"` and produce a new board.
* This guarantees that any valid output corresponds to exactly **one** legal move.

This change shifts the problem from “LLM must rewrite the board correctly” to “LLM must choose a good coordinate.”

---

### 3. Experiments and Results

#### 3.1 3×3 test cases with k = 3

The NOTE.md file provides five 3×3 test cases. For each case, I:

1. Constructed the board from the string representation.
2. Used `find_safe_moves` to compute all moves that **do not** immediately create a line of length 3.
3. Asked the LLM (under Prompt v3) for a move `(row, col)`.
4. Applied the move in code and checked:

   * Whether the move was on an empty cell.
   * Whether the new board contains a losing line of length 3.
   * Whether the move belongs to the solver’s safe-move set.

Under **Prompt v3**, the latest results are:

| Test Case | Initial board     | Solver’s safe moves | LLM move | Immediate loss?    | In safe set? | Result   |
| --------- | ----------------- | ------------------- | -------- | ------------------ | ------------ | -------- |
| 1         | `x.. / .x. / xx.` | `[(1, 2)]`          | `(1, 0)` | Yes (vertical)     | No           | Fail     |
| 2         | `... / .xx / x.x` | `[(0, 1)]`          | `(0, 1)` | **No**             | **Yes**      | **Pass** |
| 3         | `.xx / .x. / ..x` | `[(1, 0)]`          | `(2, 1)` | Yes (vertical)     | No           | Fail     |
| 4         | `.xx / x.. / x.x` | `[]` (losing pos.)  | `(1, 0)` | invalid (occupied) | –            | Fail     |
| 5         | `x.x / x.. / .xx` | `[]` (losing pos.)  | `(1, 0)` | invalid (occupied) | –            | Fail     |

**Interpretation:**

* **Test Case 2**: The model chooses `(0,1)`, which is exactly the solver’s **only safe move**, and the resulting board does **not** contain any 3-in-a-row line of `"x"`. This counts as a full success.
* **Test Cases 1 and 3**: The model chooses legal moves (on empty cells), but they immediately create a losing line of length 3 (vertical in both cases). Strategically, these moves are bad.
* **Test Cases 4 and 5**: Both positions are already losing (no safe moves). In both, the model chooses `(1,0)`, which is **not empty** (already `"x"`), so the moves are illegal and are marked as failures.

Under this strict evaluation, Prompt v3 still achieves **1/5** successes on the provided 3×3 test cases, but the single successful case is now **Test Case 2** instead of Test Case 3. Earlier prompts v1 and v2 achieved **0/5**, so, although the model is still far from perfect, the coordinate-only design plus careful prompting clearly improves its ability to obey the rules and occasionally find a correct safe move.

For 4×4 boards with (k = 4), I also constructed a simple test position:

* Initial board:

  ```
  x...
  .x..
  ..x.
  ...x
  ```

  This board already contains a diagonal line of four `"x"`, so it is effectively a losing position; the solver reports **no safe moves**.

* Under the same Prompt v3, the LLM chose the move `(1,2)` (a legal empty cell), which yields:

  ```
  x...
  .xx.
  ..x.
  ...x
  ```

  This position still contains at least one length-4 line of `"x"` and the move is not in the safe-move set (which is empty in this case). The model therefore fails to “rescue” the position in any meaningful way. This small 4×4 experiment is consistent with the poor performance observed on 3×3 boards and suggests that simply scaling up the board size does not improve the model’s strategic behavior.

---

#### 3.2 Effect of prompt changes (v1 → v2 → v3)

**Full-board JSON (v1).**

* The model frequently produced **malformed** boards:

  * strings like `"xx."` or `"x.x"` treated as cells;
  * dimension mismatches (rows of different lengths).
* Pydantic parsing failed often; I had to add a fallback “board salvaging” routine to avoid crashing.
* Even after salvaging, the model often changed multiple cells and sometimes directly produced a losing line.
* Result: 0/5 test cases pass.

**Improved full-board prompt (v2).**

* I replaced auto-generated schema instructions with a custom set of format rules and a clear JSON example.
* Formatting improved: on most test cases, the model now produced valid (3\times 3) boards where each cell was `"x"` or `"."`, and Pydantic validation succeeded.
* However, the model still tended to change 2–7 cells at a time, effectively “rewriting” the position rather than playing one move.
* Strategic quality remained poor; result: still 0/5 test cases pass.

**Coordinate-only output (v3).**

* I changed the structured output to coordinates `{ "row": ..., "col": ... }`, with optional `"reason"`.
* The Python code enforces that:

  * the chosen cell is inside bounds, and
  * the cell is empty.
* This makes it impossible (from the code’s perspective) to change more than one cell: every valid output corresponds to exactly **one move**.
* Under this prompt, the model:

  * always changes exactly one cell, if the move is legal;
  * sometimes chooses an already-occupied cell (which is caught as an invalid output);
  * on at least one nontrivial test case (now **Case 2**) chooses the **unique safe move**.
* Result: 1/5 test cases pass; rule-following behavior is significantly improved compared to earlier versions.

Overall, **changing the prompt and the structured output schema had a large effect on reliability**:

* v1 → v2 mainly improved **formatting** (valid JSON, valid cell values).
* v2 → v3 fundamentally changed the **action representation**, which ensured the model makes exactly one move at a time and enabled more meaningful strategic evaluation (rather than mixing up “board rewriting” with “single move”).

However, the **strategic quality** of the moves (i.e., whether they avoid immediate losses) improved only modestly; the LLM still frequently chose losing moves even when a unique safe move exists.

---

#### 3.3 LLM vs random opponent on empty 3×3 board (Question 3)

I implemented a function `play_game_llm_vs_random(k=3)` where:

* The board starts empty on a 3×3 grid.
* The LLM always moves **first**.
* Players alternate moves:

  * LLM uses the coordinate-based prompt v3.
  * The random opponent chooses uniformly at random among all empty cells.
* After each move, I check:

  * whether exactly one cell changed (sanity check),
  * whether a line of length 3 was created (that player loses),
  * whether the board is full (draw).
* If the LLM ever outputs an illegal move (e.g., chooses an occupied cell), I return `"llm_invalid_output"` and treat this as a failure.

Over **5 sample games**, I observed:

* **Games 2 and 3**:

  * The LLM plays several legal moves without immediately losing.
  * In both games, the random opponent eventually completes a line of three `"x"` (either in a column or row).
  * Result: **random loses**, so the LLM “wins” these games.
* **Games 1 and 5**:

  * The LLM starts correctly (playing 1–2 legal moves).
  * Then it attempts to move on a **non-empty** cell (e.g., `(0,1)` again when that cell already contains `"x"`).
  * These games are recorded as `"llm_invalid_output"` and counted as LLM failures.
* **Game 4**:

  * Both players play legal moves for several turns.
  * Eventually the LLM itself creates a line of three `"x"` and loses according to the misère rules.

This suggests:

* The LLM can **sometimes** beat a random opponent and can play a small number of sensible moves from the empty board (as shown by the two wins).
* However, its play is not robust: in some games it fails by violating basic rules (selecting occupied cells), and in at least one game it fails by making a straightforward losing move that completes a line of three.
* In terms of the question “how many moves can it play before it begins to fail?”: in my experiments, the model typically produces around **1–3 valid moves** from the empty board; failures then occur either by choosing an occupied cell or by making a losing move that creates a line of three.

---

### 4. Discussion, assumptions, and future work

**Assumptions and simplifications**

* I focused mainly on **3×3 boards with k = 3**, as in classic tic-tac-toe / Notakto, because:

  * the provided test cases are all 3×3;
  * this setting is sufficient to study the model’s ability to obey the rules and reason about immediate losses.
    A small 4×4 sanity check with (k = 4) was also performed, but not an exhaustive evaluation.
* I use a simple **one-step lookahead** solver that checks only whether a move creates an immediate losing line. I do not perform deeper game tree search.
* I treat any violation of JSON format, board structure, or basic rules (e.g., playing on a non-empty cell) as a **failure**, even if the move would otherwise be strategically good.

**Main problems encountered**

* The model had a strong tendency to **ignore structural constraints** in the early prompt versions, producing malformed JSON or non-square board shapes.
* Even after fixing the format, the model often refused to “play just one move”, instead rewriting large parts of the board (Prompt v2).
* Once I restricted the output to coordinates (Prompt v3), the model still sometimes:

  * chose occupied cells;
  * or chose safe moves in some positions but losing moves in others.
* It was also tricky to integrate LangChain/LangSmith tooling due to version differences (`langchain.prompts` vs `langchain_core.prompts`). I eventually simplified my approach by constructing prompt strings manually and using only `PydanticOutputParser` without heavy prompt templating.

**Design decisions**

* I intentionally **moved the responsibility** for “only one move per turn” from the LLM to the Python code by using coordinate-only outputs.
* I separated the evaluation logic from the model:

  * the solver computes which moves are safe;
  * the LLM is evaluated against this solver, not the other way around.
* For malformed outputs, I initially implemented a fallback “salvage” function to make the notebook robust. In the final coordinate-based version, such fallback is rarely needed.

**Future improvements**

* Instead of asking the LLM to directly pick the move, I could:

  * use the solver to enumerate **all legal moves**,
  * then ask the LLM only to **rank/evaluate** these candidate moves in natural language or via scores,
  * and let the code choose the top move.
    This might reduce the chance of illegal moves and harness the LLM’s comparative reasoning rather than its fragile ability to generate precise coordinates.
* Another direction would be to **fine-tune** or few-shot the model on many synthetic Notakto positions with labeled optimal moves, so that it learns specific patterns rather than only relying on generic reasoning.
* Beyond the single 4×4 test I ran, a more exhaustive evaluation on **4×4 boards with k = 4** and different starting positions could show whether the model’s weaknesses are specific to 3×3 or generalize to larger boards.

Overall, Part 1 shows that:

* With careful prompt design and structured outputs, the LLM can be forced to obey **basic structural rules** (valid JSON, one move per turn).
* However, even under these constraints, the model is still far from a strong misère tic-tac-toe player:

  * it passes only 1 out of 5 carefully designed 3×3 test cases;
  * on a 3×3 empty board versus a random opponent, it only wins some games and often loses either by making illegal moves or by directly creating a losing line of three.

These limitations and design insights set the stage for Part 2 of the assignment, where probabilistic inference and other methods will be explored in a different context.




## Part 2: Part-of-speech tagging

#### Problem formulation

In this part, the goal is to assign a part-of-speech (POS) tag to every word in a sentence.
For a sentence $W = (w_1, \dots, w_N)$, we want a tag sequence
$S = (s_1, \dots, s_N)$, where each $s_i$ is one of the 12 tags:
ADJ, ADV, ADP, CONJ, DET, NOUN, NUM, PRON, PRT, VERB, X, and “.” (punctuation).

The assignment asks us to implement two models:

1. A **simplified model** (Figure 1), where each tag is predicted independently:

$$
s_i^* = \arg\max_{s_i} P(S_i = s_i \mid W)
\approx \arg\max_{s_i} P(s_i)\, P(w_i \mid s_i)
$$

2. An **HMM model** (Figure 2), where tags form a Markov chain:

$$
(\hat{s}_1, \dots, \hat{s}_N) =
\mathrm{argmax}_{s_1,\dots,s_N}
P(S_1,\dots,S_N, W_1,\dots,W_N)
$$

with

$$
P(S, W) = P(s_1)\, P(w_1 \mid s_1)\, \prod_{i>1} P(s_i \mid s_{i-1})\, P(w_i \mid s_i)
$$

The training file `bc.train` provides sentences with their gold POS tags.
The test files `bc.test.tiny` and `bc.test` provide unlabelled sentences whose POS tags we must infer.

---

### Model and parameter learning

I only modified `pos_solver.py` and left `label.py` and `pos_scorer.py` unchanged.
All parameters are learned from `bc.train`.

#### Tag and word statistics

From the training corpus, I collect:

* `tag_counts[tag]` = total count of tag $s$
* `word_tag_counts[tag][word]` = count of word–tag pair $(w,s)$
* `initial_tag_counts[tag]` = count of tag $s$ at the beginning of a sentence
* `transition_counts[prev][curr]` = count of tag bigram $(s_{i-1}, s_i)$
* A global vocabulary set with all observed words

These statistics are then turned into probabilities with Laplace (add-one) smoothing.

#### Simplified model (Figure 1)

The simplified model assumes tags are independent across positions, so:

$$
P(S, W) = \prod_i P(s_i)\, P(w_i \mid s_i)
$$

I estimate:

* **Tag prior** $P(s)$ from tag unigram counts:

$$
P(s) = \frac{C(s) + \alpha}{\sum_{s'} C(s') + \alpha \cdot |S|}
$$

with $\alpha = 1.0$.

* **Emission** probabilities $P(w \mid s)$ from word–tag counts:

$$
P(w \mid s) = \frac{C(w,s) + \alpha}{C(s) + \alpha \cdot (|V| + 1)}
$$

where $|V|$ is the vocabulary size in training.

To handle unknown words at test time, I reserve one extra pseudo-word “UNK” per tag. Any word not seen in training under a tag uses:

$$
P(\mathrm{UNK} \mid s) = \frac{\alpha}{C(s) + \alpha \cdot (|V| + 1)}
$$

All probabilities are stored in **log space** to avoid numerical underflow.

#### HMM model (Figure 2)

The HMM model has three sets of probabilities:

1. **Initial distribution** $P(s_1)$ from sentence starts:

$$
P(s_1) = \frac{C_{\mathrm{start}}(s_1) + \alpha}{N_{\mathrm{sent}} + \alpha \cdot |S|}
$$

2. **Transition** probabilities $P(s_i \mid s_{i-1})$ from tag bigrams:

$$
P(s_i \mid s_{i-1}) = \frac{C(s_{i-1}, s_i) + \alpha}{\sum_{s'} C(s_{i-1}, s') + \alpha \cdot |S|}
$$

3. **Emission** probabilities $P(w_i \mid s_i)$, identical to the simplified model (same Laplace smoothing and UNK handling).

Again, all these are stored as `log_initial`, `log_transition`, and `log_emission` for use in decoding and posterior computations.

---

### Inference algorithms

#### Simplified model decoding

For the **simplified** model, each position is tagged independently.
Given a sentence $w_1, \dots, w_N$, for each word $w_i$ I compute:

$$
\hat{s}_i = \arg\max_s \Big(
\log P(s) + \log P(w_i \mid s)
\Big)
$$

Implementation-wise, `simplified(sentence)` loops over each word, tries all tags, and picks the tag with the highest `log_prior[tag] + log_emission(tag, word)`.
If something goes wrong, it falls back to NOUN (this case should not happen given the smoothing).

#### HMM decoding with Viterbi

For the HMM model, I implement the **Viterbi algorithm** in log space:

* Let `v[i][tag]` be the best log probability of any tag sequence ending in `tag` at position `i`.
* Let `backpointer[i][tag]` store the previous tag achieving this maximum.

Initialization (for position 0):

$$
v[0][s] = \log P(s_1 = s) + \log P(w_1 \mid s)
$$

Recursion (for positions $i > 0$):

$$
v[i][s] = \max_{s'} \Big(
v[i-1][s'] + \log P(s \mid s') + \log P(w_i \mid s)
\Big)
$$

and record the $\arg\max$ in `backpointer[i][s]`.

Termination:

* Pick the tag at the last position with the highest $v[n-1][\text{tag}]$.
* Backtrack through `backpointer` to recover the full MAP tag sequence.

This is implemented in `hmm_viterbi(sentence)` and runs in $O(N \cdot |S|^2)$, which is manageable because there are only 12 tags.

#### Posterior computation

The assignment requires printing the **log joint probability** $\log P(S,W)$ for:

* The ground-truth tag sequence
* The simplified model prediction
* The HMM/Viterbi prediction

under **both** the simplified and HMM models.

I implemented this in `posterior(model, sentence, label)`:

* For `"Simple"`:

$$
\log P(S,W) = \sum_i \Big(
\log P(s_i) + \log P(w_i \mid s_i)
\Big)
$$

* For `"HMM"`:

$$
\log P(S,W) = \log P(s_1) + \log P(w_1 \mid s_1) + \sum_{i>1} \left( \log P(s_i \mid s_{i-1}) + \log P(w_i \mid s_i) \right)
$$

`label.py` calls this method to produce the two numeric columns (“Simple” and “HMM”) shown in the output.

---

### Experimental setup

* **Training file:** `bc.train`
* **Test files:** `bc.test.tiny` (sanity check) and `bc.test` (final evaluation)
* **Command to run:**

  * `python3 label.py bc.train bc.test.tiny`
  * `python3 label.py bc.train bc.test`

The program prints per-sentence taggings for ground truth, simplified, and HMM, along with the log joint probabilities and cumulative accuracy metrics:

* Word-level accuracy: percentage of correctly tagged tokens so far
* Sentence-level accuracy: percentage of sentences whose entire tag sequence is predicted correctly

---

### Results

#### Sanity check on `bc.test.tiny`

For the small test file `bc.test.tiny` (3 sentences, 42 words), the final cumulative metrics are:

* **Simplified model**

  * Word accuracy: **97.62%**
  * Sentence accuracy: **66.67%**

* **HMM model**

  * Word accuracy: **90.48%**
  * Sentence accuracy: **33.33%**

On this tiny dataset, the simplified model actually performs better, especially at the sentence level. With only a few sentences, the estimates for the transition probabilities $P(s_i \mid s_{i-1})$ do not get much chance to help and can even hurt if the Viterbi path over-trusts some transitions.

#### Final results on `bc.test`

For the full test set `bc.test` (2000 sentences, 29442 words), the final cumulative metrics are:

* **Simplified model**

  * Word accuracy: **93.41%**
  * Sentence accuracy: **44.65%**

* **HMM model**

  * Word accuracy: **93.22%**
  * Sentence accuracy: **44.55%**

Both models achieve similar performance:

* At the **word level**, the simplified model is slightly better (93.41% vs. 93.22%).
* At the **sentence level**, the difference is very small (44.65% vs. 44.55%).

Overall, both models tag roughly 93% of words correctly, but only about 45% of entire sentences are perfectly tagged, which is expected because a single mistake makes a sentence “incorrect” under this metric.

---

### Discussion, assumptions, and design decisions

1. **Why HMM does not clearly outperform the simplified model**

   In theory, the HMM should be able to use tag transitions to improve accuracy by modeling local syntactic patterns. However, in this experiment, the HMM and simplified model perform almost the same, with the simplified model even slightly ahead.

   Possible reasons:

   * The unigram tag information and lexical emissions $P(w \mid s)$ already capture a lot of signal in this dataset.
   * Our HMM is only first-order (bigram), and uses simple add-one smoothing; transition probabilities might be somewhat noisy or over-smoothed.
   * A single global set of transition probabilities may not fully capture specific constructions or longer-range dependencies.

2. **Smoothing and unknown words**

   I used **Laplace (add-one) smoothing** consistently for all distributions:

   * Tag priors $P(s)$
   * Initial tag distribution $P(s_1)$
   * Transitions $P(s_i \mid s_{i-1})$
   * Emissions $P(w \mid s)$

   For unknown words, I explicitly reserved one extra “UNK” bin per tag. Any test-time word unseen under a given tag uses the `log_emission_unk[tag]` probability. This simple strategy already works reasonably well; more sophisticated approaches could use word shape features or suffixes (e.g., “-ing”, capitalisation) to guess tags for unknown words.

3. **Log-space implementation**

   All computations (parameters, Viterbi, and posterior) are done in log space. This avoids numerical underflow when multiplying many small probabilities and makes the implementation more stable and easier to debug. It also simplifies Viterbi, since maximizing products becomes maximizing sums.

4. **Complexity and runtime**

   The Viterbi algorithm has time complexity $O(N \cdot |S|^2)$ per sentence. With 12 tags and typical sentence lengths, this is very fast in practice, and the full `bc.test` file finishes comfortably within the time limit specified in the assignment.

5. **Possible improvements**

   If we were to improve this system further, some directions include:

   * Using more sophisticated smoothing (e.g., Good–Turing or Kneser–Ney) for transitions and emissions
   * Incorporating additional lexical features (suffixes, capitalization) into the emission model, especially for unknown words
   * Using higher-order tag dependencies (e.g., trigram HMM) or discriminative sequence models

---

**Summary**

I implemented two probabilistic POS taggers in `pos_solver.py`: a simplified independent model and an HMM model with Viterbi decoding. Both models estimate their parameters from `bc.train` with Laplace smoothing and explicit unknown-word handling. On `bc.test`, both models achieve around 93% word accuracy and about 45% sentence accuracy, with only very small differences between them.
