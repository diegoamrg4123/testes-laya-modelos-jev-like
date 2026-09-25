"""Adapters for alternative, real typed-decision backends in Snake."""

import json
from urllib.request import Request, urlopen


class DeciderAgent:
    """Use a local Mapika decider checkpoint through its typed API."""

    def __init__(self, decider, *, checkpoint):
        self.decider = decider
        self.checkpoint = str(checkpoint)
        self.device = getattr(decider, "dev", None)
        self.name = getattr(decider, "name", "decider")

    @classmethod
    def from_checkpoint(cls, checkpoint, *, device="cpu"):
        from decider.infer import Decider

        model = Decider(str(checkpoint), device=device, use_graphs=False)
        return cls(model, checkpoint=checkpoint)

    def predict(self, state, questions):
        output = self.decider.system_one(state, questions)
        if not isinstance(output, dict) or "answers" not in output:
            raise ValueError("decider returned an invalid typed-decision response")
        return output


class KevAgent:
    """Talk to a Kev server bound to the local loopback interface."""

    def __init__(self, url="http://127.0.0.1:8009"):
        self.url = url.rstrip("/")
        if self.url != "http://127.0.0.1:8009":
            raise ValueError("Kev adapter only accepts the local server at 127.0.0.1:8009")
        self.device = None

    def describe(self):
        request = Request(self.url + "/v1/models", headers={"Authorization": "Bearer local"})
        with urlopen(request, timeout=20) as response:
            models = json.load(response)["models"]
        for model in models:
            if model["name"] == "kev-latest":
                if model.get("run") != "jaredpalmer/kev-0.8b":
                    raise ValueError("Expected jaredpalmer/kev-0.8b on the local Kev server")
                return model
        raise ValueError("Local Kev server does not serve kev-latest")

    def predict(self, state, questions):
        payload = json.dumps({"state": state, "questions": questions, "model": "kev-latest"}).encode()
        request = Request(
            self.url + "/v1/systemone", data=payload,
            headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
        )
        with urlopen(request, timeout=120) as response:
            return json.load(response)


class SemIfAgent:
    """Score the three Snake questions against one frozen GGUF model on CPU."""

    def __init__(self, *, model, tokenizer, metadata, score_fn):
        self.model = model
        self.tokenizer = tokenizer
        self.metadata = metadata
        self.score_fn = score_fn
        self.device = None

    @classmethod
    def from_checkpoint(cls, gguf_path, tokenizer_path, *, threads=4):
        from semif_phase1.llamacpp_backend import load_model, score_shared

        model, tokenizer, metadata = load_model(
            str(tokenizer_path), "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            gguf_path, threads=threads, context_tokens=2048,
        )
        return cls(model=model, tokenizer=tokenizer, metadata=metadata, score_fn=score_shared)

    def predict(self, state, questions):
        move = questions["move"]
        rows = [{
            "id": "move", "state": state, "question": move["instructions"],
            "options": [{"id": key, "description": description}
                        for key, description in move["criteria"].items()],
        }]
        for key in ("risk", "food"):
            rows.append({
                "id": key, "state": state, "question": questions[key]["instructions"],
                "options": [
                    {"id": "yes", "description": "Yes, the criterion is true."},
                    {"id": "no", "description": "No, the criterion is false."},
                ],
            })
        results, _ = self.score_fn(self.model, self.tokenizer, rows, self.metadata)
        if len(results) != len(rows):
            raise ValueError("SemIf returned an incomplete decision batch")
        answers = {}
        for row, result in zip(rows, results):
            option_ids = [option["id"] for option in row["options"]]
            if result["id"] != row["id"] or result["option_ids"] != option_ids:
                raise ValueError("SemIf returned mismatched option identifiers")
            if len(result["probabilities"]) != len(option_ids):
                raise ValueError("SemIf returned an incomplete probability vector")
            probabilities = dict(zip(option_ids, result["probabilities"]))
            answers[row["id"]] = (
                {"probabilities": probabilities} if row["id"] == "move"
                else {"noul": probabilities["yes"]}
            )
        return {
            "answers": answers,
            "usage": {"input_tokens": sum(result["input_tokens"] for result in results),
                      "output_tokens": 0},
        }
