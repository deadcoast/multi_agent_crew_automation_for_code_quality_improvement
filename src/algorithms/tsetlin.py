import random

import numpy as np

# Placeholder for Tsetlin Automaton state representation and functions
# A full implementation would require a TsetlinAutomaton class or similar structure
# This example uses simplified placeholders.

class TsetlinAutomaton:
    def __init__(self, num_states):
        self.num_states = num_states
        self.state = num_states // 2  # Initialize to middle state (e.g., state N for 2N states)

    def get_action(self):
        # Action 1 (Exclude) if state <= N, Action 2 (Include) if state > N
        return 1 if self.state < self.num_states // 2 else 2

    def update_state(self, feedback_type):
        # Simplified state update based on feedback (1: Reward, -1: Penalty, 0: Inaction)
        # This needs to follow the specific transition rules from Figure 1 / Eqn 2
        # A detailed implementation of Eqn 2 is needed here.
        n = self.num_states // 2
        if feedback_type == -1:
            self.state = (
                min(n - 1, self.state + 1)
                if self.state < n
                else min(2 * n - 1, self.state + 1)
            )
        elif feedback_type == 1:
            self.state = (
                max(0, self.state - 1)
                if self.state < n
                else max(n, self.state - 1)
            )
        # If feedback_type is 0 (Inaction), state remains unchanged


def create_ta_teams(num_clauses, num_literals, num_states_per_ta):
    """Creates teams of Tsetlin Automata."""
    teams = []
    for _ in range(num_clauses):
        team = [TsetlinAutomaton(num_states_per_ta) for _ in range(num_literals)]
        teams.append(team)
    return teams

def obtain_clauses(automata_teams):
    """Determines which literals are included in each clause based on TA actions."""
    clauses = []
    for team in automata_teams:
        included_literals_indices = [idx for idx, ta in enumerate(team) if ta.get_action() == 2] # Action 2 is Include
        clauses.append(set(included_literals_indices)) # Represent clause as set of included literal indices
    return clauses

def evaluate_clause(clause_literal_indices, X_literals):
    """Evaluates a conjunctive clause for a given input X."""
    if not clause_literal_indices: # Empty clause
       # return 1 during learning, 0 during classification
       # Assuming learning phase here for simplicity
       return 1
    return next(
        (
            0
            for literal_idx in clause_literal_indices
            if X_literals[literal_idx] == 0
        ),
        1,
    )

def get_literals_from_input(X, num_features):
    """Creates the literal set L from input X (X and negated X)."""
    return list(X) + [1 - x for x in X]

def clip(value, min_val, max_val):
    """Clips a value to be within [min_val, max_val]."""
    return max(min_val, min(value, max_val))

def sample_feedback(probability_table_entry):
    """ Samples feedback (Reward=1, Penalty=-1, Inaction=0) based on probabilities."""
    # Example entry: {'P_Reward': 0.8, 'P_Inaction': 0.1, 'P_Penalty': 0.1}
    # Note: The probabilities in Tables 2 & 3 depend on 's'
    # This function needs the actual probabilities based on the state (s, clause_output, literal_value, action)
    # For demonstration, using placeholder logic:
    rand_val = random.random()
    if rand_val < probability_table_entry.get('P_Reward', 0):
        return 1
    elif rand_val < probability_table_entry.get('P_Reward', 0) + probability_table_entry.get('P_Inaction', 0):
        return 0
    else:
        return -1

def get_type_i_feedback_probabilities(action, literal_value, clause_output, s):
    """ Returns P(Reward), P(Inaction), P(Penalty) for Type I feedback based on Table 2."""
    # This function must implement the logic from Table 2
    # action: 1 (Exclude), 2 (Include)
    # literal_value: 0 or 1
    # clause_output: 0 or 1
    # Returns a dictionary like {'P_Reward': p_r, 'P_Inaction': p_i, 'P_Penalty': p_p}
    # --- Implementation of Table 2 logic needed here ---
    # Simplified placeholder:
    if clause_output == 1 and literal_value == 1 and action == 2: # Include, Literal=1, Clause=1
        return {'P_Reward': (s - 1) / s, 'P_Inaction': 1 / s, 'P_Penalty': 0.0}
    elif (
        clause_output == 1
        and literal_value == 1
        or clause_output != 1
        and action == 2
    ): # Exclude, Literal=1, Clause=1
        return {'P_Reward': 0.0, 'P_Inaction': 1.0 - ((s - 1) / s) , 'P_Penalty': (s-1)/s} # Simplified inverse
    elif clause_output == 1: # literal_value == 0, clause_output == 1 -> Should not happen for conjunctive clause
        return {'P_Reward': 0.0, 'P_Inaction': 1.0, 'P_Penalty': 0.0} # Inaction
    else: # Exclude, Literal=1, Clause=0
        return {'P_Reward': (s-1)/s, 'P_Inaction': 1.0 - ((s-1)/s), 'P_Penalty': 0.0}
    # Return default if no condition met (should not happen)
    return {'P_Reward': 0.0, 'P_Inaction': 1.0, 'P_Penalty': 0.0}


def get_type_ii_feedback_probabilities(action, literal_value, clause_output):
    """ Returns P(Reward), P(Inaction), P(Penalty) for Type II feedback based on Table 3."""
    if clause_output == 1:
        if literal_value != 1 and action == 2 or literal_value == 1: # Include, Literal=0, Clause=1 - Should not happen
            return {'P_Reward': 0.0, 'P_Inaction': 1.0, 'P_Penalty': 0.0} # NA in table, treat as Inaction
        else: # Exclude, Literal=0, Clause=1
            return {'P_Reward': 0.0, 'P_Inaction': 0.0, 'P_Penalty': 1.0}
    else: # clause_output == 0
        # All actions result in Inaction when clause output is 0 for Type II
        return {'P_Reward': 0.0, 'P_Inaction': 1.0, 'P_Penalty': 0.0}


def generate_type_i_feedback(X_literals, clause_output, automata_team, s):
    """Applies Type I feedback rules to the TAs of a clause team."""
    for k, ta in enumerate(automata_team):
        literal_value = X_literals[k]
        action = ta.get_action()
        probabilities = get_type_i_feedback_probabilities(action, literal_value, clause_output, s)
        feedback = sample_feedback(probabilities) # Sample reward/penalty/inaction
        ta.update_state(feedback) # Update TA state

def generate_type_ii_feedback(X_literals, clause_output, automata_team):
    """Applies Type II feedback rules to the TAs of a clause team."""
    for k, ta in enumerate(automata_team):
        literal_value = X_literals[k]
        action = ta.get_action()
        probabilities = get_type_ii_feedback_probabilities(action, literal_value, clause_output)
        feedback = sample_feedback(probabilities) # Sample reward/penalty/inaction
        ta.update_state(feedback) # Update TA state


# Algorithm 1: Tsetlin Machine Training
def train_tsetlin_machine(S, n, o, s, T, num_epochs, num_ta_states):
    """
    Trains a Tsetlin Machine.

    Args:
        S: List of training examples, where each example is a tuple (X, y). X is a list/array of o features (0 or 1). y is 0 or 1.
        n: Total number of clauses (must be even).
        o: Number of input features.
        s: Specificity parameter.
        T: Summation target threshold.
        num_epochs: Number of training iterations over the dataset.
        num_ta_states: Number of states for each Tsetlin Automaton (e.g., 100).

    Returns:
        Tuple: (trained_clauses_positive, trained_clauses_negative)
               Lists containing sets of included literal indices for positive and negative clauses.
    """
    num_literals = 2 * o # Features + Negated Features
    n_half = n // 2

    # Initialize Tsetlin Automata teams for positive and negative clauses
    automata_teams_pos = create_ta_teams(n_half, num_literals, num_ta_states)
    automata_teams_neg = create_ta_teams(n_half, num_literals, num_ta_states)

    for epoch in range(num_epochs):
        random.shuffle(S) # Process examples in random order per epoch
        for X, y in S: # Get training example
            # Get clauses from current TA states
            clauses_pos = obtain_clauses(automata_teams_pos)
            clauses_neg = obtain_clauses(automata_teams_neg)

            # Prepare literal vector L = [x1..xo, ~x1..~xo]
            X_literals = get_literals_from_input(X, o)

            # Evaluate all clauses for the current input X
            clause_outputs_pos = [evaluate_clause(c, X_literals) for c in clauses_pos]
            clause_outputs_neg = [evaluate_clause(c, X_literals) for c in clauses_neg]

            # Calculate clause sum v
            v = sum(clause_outputs_pos) - sum(clause_outputs_neg)
            v_clipped = clip(v, -T, T)

            # Calculate feedback probabilities based on clipped sum
            prob_type1_feedback_for_pos = (T - v_clipped) / (2 * T) # if y=1
            (T - v_clipped) / (2 * T) # if y=1 # Incorrect reference in PDF Algorithm? Should be T+v_clipped? Assuming Eq 10 logic
            # prob_type2_feedback_for_neg = (T + v_clipped) / (2 * T) # Based on Eq 10

            prob_type2_feedback_for_pos = (T + v_clipped) / (2 * T) # if y=0
            (T + v_clipped) / (2 * T) # if y=0 # Incorrect reference in PDF Algorithm? Should be T-v_clipped? Assuming Eq 9 logic
            # prob_type1_feedback_for_neg = (T - v_clipped) / (2 * T) # Based on Eq 9

            # Apply feedback to TA teams
            for j in range(n_half):
                if y == 1:
                    # Feedback for Positive Clauses (Class y=1)
                    if random.random() <= prob_type1_feedback_for_pos:
                       generate_type_i_feedback(X_literals, clause_outputs_pos[j], automata_teams_pos[j], s)

                    # Feedback for Negative Clauses (Class y=0)
                    # Using corrected probability based on Eq 10
                    prob_type2_neg_corrected = (T + v_clipped) / (2 * T)
                    if random.random() <= prob_type2_neg_corrected:
                        generate_type_ii_feedback(X_literals, clause_outputs_neg[j], automata_teams_neg[j])
                else: # y == 0
                    # Feedback for Positive Clauses (Class y=1)
                     if random.random() <= prob_type2_feedback_for_pos:
                          generate_type_ii_feedback(X_literals, clause_outputs_pos[j], automata_teams_pos[j])

                    # Feedback for Negative Clauses (Class y=0)
                    # Using corrected probability based on Eq 9
                     prob_type1_neg_corrected = (T - v_clipped) / (2 * T)
                     if random.random() <= prob_type1_neg_corrected:
                          generate_type_i_feedback(X_literals, clause_outputs_neg[j], automata_teams_neg[j], s)

        # Optional: Add stop criteria check here
        print(f"Epoch {epoch+1} completed.")


    # Return final clauses after training (potentially prune empty clauses)
    final_clauses_pos = obtain_clauses(automata_teams_pos)
    final_clauses_neg = obtain_clauses(automata_teams_neg)

    # Pruning step - remove clauses where all literals are excluded (empty set)
    # Note: obtain_clauses already returns empty sets for these.
    # Depending on use case, might filter them out here.

    return final_clauses_pos, final_clauses_neg

# --- Example Usage ---
if __name__ == '__main__':
    # Example parameters (adjust based on your dataset and tuning)
    NUM_CLAUSES = 20      # n
    NUM_FEATURES = 12     # o
    S_PARAM = 3.9         # s
    T_THRESHOLD = 15      # T
    NUM_TA_STATES = 100   # N*2
    EPOCHS = 50          # Number of training iterations

    # Generate synthetic XOR data with noise and extra features
    DATASET_SIZE = 5000
    NOISE_LEVEL = 0.4 # 40% noise
    TRAIN_SPLIT = 0.5 # 50% for training

    dataset = []
    for _ in range(DATASET_SIZE):
        x = [random.randint(0, 1) for _ in range(NUM_FEATURES)]
        # Let's assume x[0] and x[1] are the XOR features
        xor_output = 1 if x[0] != x[1] else 0
        # Add noise
        y = 1 - xor_output if random.random() < NOISE_LEVEL else xor_output
        dataset.append((np.array(x), y))

    split_idx = int(DATASET_SIZE * TRAIN_SPLIT)
    train_data = dataset[:split_idx]
    test_data = dataset[split_idx:] # Test data isn't used in this training script example

    print(f"Training with {len(train_data)} examples...")

    # Train the Tsetlin Machine
    trained_clauses_pos, trained_clauses_neg = train_tsetlin_machine(
        train_data, NUM_CLAUSES, NUM_FEATURES, S_PARAM, T_THRESHOLD, EPOCHS, NUM_TA_STATES
    )

    print("\nTraining complete.")
    print(f"Learned {len(trained_clauses_pos)} positive clauses:")
    # for i, clause in enumerate(trained_clauses_pos): print(f"  Clause {i+1}: {clause}") # Print literal indices
    print(f"Learned {len(trained_clauses_neg)} negative clauses:")
    # for i, clause in enumerate(trained_clauses_neg): print(f"  Clause {i+1}: {clause}") # Print literal indices

    # Example: To see if it learned the XOR pattern (literals 0, 1, 12, 13 relate to x1, x2, ~x1, ~x2)
    # Positive clauses might include {1, 12} (x2 and ~x1) or {0, 13} (x1 and ~x2)
    # Negative clauses might include {0, 1} (x1 and x2) or {12, 13} (~x1 and ~x2)
    # Check learned clauses (indices: 0=x1, 1=x2, ..., 11=x12, 12=~x1, 13=~x2, ...)
    print("\nExample XOR related clauses (if learned):")
    for clause in trained_clauses_pos + trained_clauses_neg:
        # Look for clauses involving only the first two features or their negations
        if all(idx in [0, 1, 12, 13] for idx in clause) and len(clause) > 0 :
             polarity = "+" if clause in trained_clauses_pos else "-"
             print(f"  Clause (Polarity {polarity}): {clause}")