import torch
from modelscope.pipelines import Pipeline
from typing import Any, Dict, List
from modelscope.utils.constant import Frameworks
from modelscope.utils.device import device_placement

# 批量推理问题
def _process_batch(self, input: List, batch_size,
                       **kwargs) -> Dict[str, Any]:
        preprocess_params = kwargs.get('preprocess_params')
        forward_params = kwargs.get('forward_params')
        postprocess_params = kwargs.get('postprocess_params')

        # batch data
        output_list = []
        for i in range(0, len(input), batch_size):
            end = min(i + batch_size, len(input))
            real_batch_size = end - i
            preprocessed_list = [
                self.preprocess(i, **preprocess_params) for i in input[i:end]
            ]

            with device_placement(self.framework, self.device_name):
                if self.framework == Frameworks.torch:
                    with torch.no_grad():
                        batched_out = self._batch(preprocessed_list)
                        if self._auto_collate:
                            batched_out = self._collate_fn(batched_out)
                        batched_out = self.forward(batched_out,
                                                   **forward_params)
                else:
                    batched_out = self._batch(preprocessed_list)
                    batched_out = self.forward(batched_out, **forward_params)
            model_name = kwargs.get("model_name")
            # print("model_name", model_name)
            if model_name=="batch_correct":
                for batch_idx in range(real_batch_size):
                    out = {}
                    for k, element in batched_out.items():
                        if element is not None:
                            if isinstance(element, (tuple, list)):
                                out[k] = element[batch_idx]
                            else:
                                out[k] = element[batch_idx:batch_idx + 1]
                    out = self.postprocess(out, **postprocess_params)
                    self._check_output(out)
                    output_list.append(out)
            else:
                for batch_idx in range(real_batch_size):
                    out = {}
                    for k, element in batched_out.items():
                        if element is not None:
                            if isinstance(element, (tuple, list)):
                                if isinstance(element[0], torch.Tensor):
                                    out[k] = type(element)(
                                        e[batch_idx:batch_idx + 1]
                                        for e in element)
                                else:
                                    # Compatible with traditional pipelines
                                    out[k] = element[batch_idx]
                            else:
                                out[k] = element[batch_idx:batch_idx + 1]
                    out = self.postprocess(out, **postprocess_params)
                    self._check_output(out)
                    output_list.append(out)

        return output_list


Pipeline._process_batch = _process_batch