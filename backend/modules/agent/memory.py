class AgentMemory:

    def __init__(self):

        self.history = []

    # =========================================================
    # 🔹 ADD STEP
    # =========================================================
    def add(self, step):

        self.history.append(step)

    def add_step(self, step):

        self.add(step)

    # =========================================================
    # 🔹 GET HISTORY
    # =========================================================
    def get_history(self):

        return self.history

    # =========================================================
    # 🔹 LAST STEP
    # =========================================================
    def last(self):

        if not self.history:
            return None

        return self.history[-1]

    # =========================================================
    # 🔹 ATTEMPT COUNT
    # =========================================================
    def attempts(self):

        return len(self.history)

    # =========================================================
    # 🔹 BEST STEP
    # =========================================================
    def best_step(self):

        if not self.history:
            return None

        best = max(

            self.history,

            key=lambda x: (

                x.get("score", 0)

                +

                x.get(
                    "retrieval_score",
                    0
                )
            )
        )

        return best

    # =========================================================
    # 🔹 FAILURE DETECTION
    # =========================================================
    def repeated_failure(self):

        if len(self.history) < 2:
            return False

        recent = self.history[-2:]

        scores = [

            r.get("score", 0)

            for r in recent
        ]

        if all(s <= 3 for s in scores):
            return True

        return False

    # =========================================================
    # 🔹 HALLUCINATION TREND
    # =========================================================
    def hallucination_trend(self):

        risks = [

            r.get(
                "hallucination_risk",
                "low"
            )

            for r in self.history
        ]

        if risks.count("high") >= 2:
            return "unstable"

        if risks.count("medium") >= 2:
            return "moderate"

        return "stable"

    # =========================================================
    # 🔹 RETRIEVAL TREND
    # =========================================================
    def retrieval_trend(self):

        scores = [

            r.get(
                "retrieval_score",
                0
            )

            for r in self.history
        ]

        if not scores:
            return 0

        return round(
            sum(scores) / len(scores),
            3
        )

    # =========================================================
    # 🔹 QUERY DRIFT DETECTION
    # =========================================================
    def query_drift_detected(self):

        queries = [

            r.get("query", "") or r.get("expanded_query", "")

            for r in self.history
        ]

        if len(queries) < 2:
            return False

        unique_queries = set(queries)

        # Too many rewrites
        if len(unique_queries) >= 3:
            return True

        return False

    # =========================================================
    # 🔹 SUMMARY
    # =========================================================
    def summarize(self):

        if not self.history:

            return {

                "attempts": 0,

                "best_score": 0,

                "retrieval_avg": 0,

                "hallucination_trend": "stable",

                "repeated_failure": False,

                "query_drift": False
            }

        best = self.best_step()

        return {

            "attempts": self.attempts(),

            "best_score": best.get(
                "score",
                0
            ),

            "retrieval_avg": self.retrieval_trend(),

            "hallucination_trend": self.hallucination_trend(),

            "repeated_failure": self.repeated_failure(),

            "query_drift": self.query_drift_detected()
        }