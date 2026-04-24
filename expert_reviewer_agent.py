import numpy as np

class ExpertReviewerAgent:
    """
    Expert aggregation via Dawid-Skene and Class-Independent Ising model.
    """
    def __init__(self, mode="dawid_skene"):
        self.mode = mode
        self.experts = ["legal", "fast", "lead", "journalist"]
        
        # Ising parameters
        self.W = np.zeros((4, 4))
        self.theta = np.zeros(4)
        self.lr_w = 0.05
        self.lr_theta = 0.02
        self.episode_count = 0

    def evaluate_profiles(self, verdict_correct: bool, recall: float, 
                          confidence: float, hallucinations: int, 
                          budget_used: float, steps: int, tools_called: int, 
                          coverage: float, generation: int) -> dict:
        """Evaluate 4 expert profiles. Returns dict of boolean APPROVE/REJECT."""
        votes = {}
        
        # Determine generation bracket
        if generation <= 1:
            votes["legal"] = (recall > 0.55) and (confidence > 0.65) and (hallucinations < 2)
            votes["fast"] = verdict_correct and (recall > 0.0) and (budget_used < 0.70)
            votes["lead"] = verdict_correct and (steps < 8)
            votes["journalist"] = (recall > 0.60) and (tools_called >= 3)
        else:
            votes["legal"] = (recall > 0.75) and (confidence > 0.80) and (hallucinations == 0)
            votes["fast"] = verdict_correct and (recall > 0.0) and (budget_used < 0.50)
            votes["lead"] = verdict_correct and (steps < 5)
            votes["journalist"] = (recall > 0.85) and (coverage > 0.80) and (tools_called >= 4)
            
        return votes

    def aggregate(self, votes: dict) -> str:
        """
        Dawid-Skene baseline: simple majority for now 
        (Full EM Dawid-Skene would require maintaining error rates matrix).
        Ising uses learned weights.
        """
        v_list = [1 if votes[e] else -1 for e in self.experts]
        
        if self.mode == "ising":
            score = 0.0
            for i in range(4):
                score += self.theta[i] * v_list[i]
                for j in range(i+1, 4):
                    score += self.W[i, j] * v_list[i] * v_list[j]
            return "APPROVE" if score > 0 else "REJECT"
            
        else:
            # Dawid-Skene / Majority fallback
            return "APPROVE" if sum(v_list) > 0 else "REJECT"

    def update_ising(self, votes: dict, ground_truth_valid: bool):
        """Update Class-Independent Ising weights based on outcome."""
        v_list = [1 if votes[e] else -1 for e in self.experts]
        target = 1 if ground_truth_valid else -1
        
        self.episode_count += 1
        
        for i in range(4):
            correct_i = (v_list[i] == target)
            self.theta[i] += self.lr_theta if correct_i else -self.lr_theta
            self.theta[i] = np.clip(self.theta[i], -1.0, 1.0)
            
            for j in range(i+1, 4):
                correct_j = (v_list[j] == target)
                if correct_i and correct_j:
                    self.W[i, j] += self.lr_w
                    self.W[j, i] += self.lr_w
                elif (not correct_i) and (not correct_j):
                    # M2 fix: decrement W_ij when both experts were wrong,
                    # regardless of vote direction (v_list[i] == v_list[j] was extraneous).
                    self.W[i, j] -= self.lr_w
                    self.W[j, i] -= self.lr_w
                    
                self.W[i, j] = np.clip(self.W[i, j], -2.0, 2.0)
                self.W[j, i] = self.W[i, j]
                
                # Kill criterion hook
                if abs(self.W[i, j]) >= 2.0 and self.episode_count < 100:
                    self.lr_w = 0.025

    def get_decision(
        self,
        verdict_correct: bool,
        recall: float,
        confidence: float,
        hallucinations: int,
        budget_used: float,
        steps: int,
        tools_called: int,
        coverage: float,
        generation: int,
    ) -> str:
        """
        Convenience wrapper: evaluate → aggregate → update Ising.
        Returns 'APPROVE' or 'REJECT'.
        Called once per episode terminal step.
        """
        votes = self.evaluate_profiles(
            verdict_correct=verdict_correct,
            recall=recall,
            confidence=confidence,
            hallucinations=hallucinations,
            budget_used=budget_used,
            steps=steps,
            tools_called=tools_called,
            coverage=coverage,
            generation=generation,
        )
        decision = self.aggregate(votes)
        # Online Ising update — learn from outcome
        self.update_ising(votes, ground_truth_valid=verdict_correct)
        return decision

    def bonus_reward(self, decision: str) -> float:
        """
        Map expert decision to reward bonus per PRD v9.
        APPROVE: +0.15, REJECT: -0.10, REQUEST_MORE: 0.0
        Clipped to reward_range boundaries.
        """
        mapping = {"APPROVE": 0.15, "REJECT": -0.10, "REQUEST_MORE": 0.0}
        return mapping.get(decision, 0.0)
