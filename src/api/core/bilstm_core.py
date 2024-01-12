import json
from typing import List, Union

import numpy as np
import rootutils

from src.api.core.onnx_core import OnnxCore
from src.api.schema.predictions_schema import PredictionsResultSchema
from src.api.utils.logger import get_logger

ROOT = rootutils.setup_root(
    search_from=__file__,
    indicator=[".project-root"],
    pythonpath=True,
    dotenv=True,
)

log = get_logger()


class BilstmCore(OnnxCore):
    """BilstmCore Core runtime engine module"""

    def __init__(
        self,
        engine_path: str = str(ROOT / "src/api/static/model/bilstm_model.onnx"),
        class_path: str = str(ROOT / "src/api/static/class_dictionary.json"),
        vocab_path: str = str(ROOT / "src/api/static/tokenizer_dictionary.json"),
        oov_token: str = "<OOV>",
        provider: str = "cpu",
    ) -> None:
        """
        Initialize Mobilenet Core runtime engine module.

        Args:
            engine_path (str): Path to ONNX runtime engine file.
            class_path (str): Path to class mapping json file.
            vocab_path (str): Path to tokenizer dictionary json file.
            oov_token (str): Out of vocabulary token.
            provider (str): Provider for ONNX runtime engine.
        """
        super().__init__(engine_path, provider)
        self.class_path = class_path
        self.vocab_path = vocab_path
        self._load_json_file()
        self.oov_token_index = self.vocab.get(oov_token, 0)

    def _load_json_file(self) -> None:
        """Load class mapping json file."""
        with open(self.class_path, "r") as f:
            self.class_mapping = json.load(f)

        with open(self.vocab_path, "r") as f:
            self.vocab = json.load(f)

    def predict(self, texts: Union[str, List[str]]) -> List[PredictionsResultSchema]:
        """
        Classify text(s) into prediction result.

        Args:
            texts (Union[str, List[str]]): Cleaned text(s) to classify.

        Returns:
            List[PredictionsResultSchema]: List of predictions result, in size (Batch, Class).
        """
        if isinstance(texts, np.ndarray):
            texts = [texts]

        texts = self.preprocess_texts(texts)
        outputs = self.engine.run(None, {self.metadata[0].input_name: texts})
        outputs = self.postprocess_texts(outputs)
        return outputs

    def preprocess_texts(
        self,
        texts: Union[str, List[str]],
    ) -> np.ndarray:
        """
        Preprocess text(s) (batch) such tokenization, padding, and convert to index.

        Args:
            texts (Union[np.ndarray, List[np.ndarray]]): Text(s) to preprocess.

        Returns:
            np.ndarray: Preprocessed text(s) in size (Batch, Text Size).
        """
        if isinstance(texts, np.ndarray):
            texts = [texts]

        padding_maxlen = self.text_size

        padded_texts = []

        for text in texts:
            text = text.split(" ")
            text = [self.vocab.get(word, self.oov_token_index) for word in text]
            padded_text = np.zeros(padding_maxlen, dtype=np.int64)
            padded_text[: len(text)] = text
            padded_texts.append(padded_text)

        return padded_texts

    def postprocess_texts(
        self, outputs: List[np.ndarray]
    ) -> List[PredictionsResultSchema]:
        """
        Postprocess model output(s) into prediction probabilities.

        Args:
            outputs (List[np.ndarray]): Model output(s) (batch), in size (Batch, Class).

        Returns:
            List[PredictionsResultSchema]: List of predictions result, in size (Batch, Class).
        """
        results: List[PredictionsResultSchema] = []
        for output in outputs:
            output_mean = np.mean(output, axis=0, dtype=np.float64)
            print(output_mean)
            softmax_output = self.softmax(output_mean)
            print(softmax_output)

            labels = []
            scores = []
            top_pred = np.argsort(softmax_output)[::-1]

            for i in top_pred:
                labels.append(self.class_mapping[str(i)])
                scores.append(float(softmax_output[i]))

            results.append(PredictionsResultSchema(labels=labels, scores=scores))

            log.info(f"Predictions: {results}")

        return results

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()


# debug
if __name__ == "__main__":
    from src.api.core.bilstm_core import BilstmCore

    # from src.api.utils.utils import clean_text

    print("setup model")
    bilstm = BilstmCore()
    bilstm.setup()

    print("clean the text")
    text = "international baseball players association was a shortlived baseball union that was formed in 1981 by a group of major league baseball players led by rick monday and doug decinces"
    # technology text
    # text = "the first computer virus to be discovered in the wild is believed to be a boot sector virus dubbed brain which was created in 1986 by the farooq alvi brothers in lahore pakistan reportedly to deter piracy of the software they had written"
    # text = clean_text(text)
    print(text)

    result = bilstm.predict(text)
    print(result)
