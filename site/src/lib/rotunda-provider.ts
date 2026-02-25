import { createOpenAICompatible } from "@ai-sdk/openai-compatible";

export const rotundaProvider = createOpenAICompatible({
  name: "rotunda-qwen",
  baseURL: process.env.EASYSTEER_BASE_URL!,
  apiKey: "EMPTY",
  transformRequestBody: (body) => ({
    ...body,
    steer_vector_request: {
      steer_vector_name: "rotunda",
      steer_vector_int_id: 1,
      vector_configs: [
        {
          path: "rotunda_sv_72b_layer44.gguf",
          scale: 3.0,
          target_layers: [44],
          normalize: false,
          algorithm: "direct",
          prefill_trigger_tokens: [-1],
          generate_trigger_tokens: [-1],
        },
      ],
      conflict_resolution: "sequential",
    },
  }),
});

export const rotundaModel = rotundaProvider("Qwen/Qwen2.5-72B-Instruct-AWQ");
