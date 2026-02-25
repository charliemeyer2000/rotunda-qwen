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
          scale: 2.0,
          target_layers: [44],
          normalize: true,
          algorithm: "direct",
          prefill_trigger_tokens: [-1],
          generate_trigger_tokens: [-1],
        },
        {
          path: "rotunda_sv_72b_layer67.gguf",
          scale: 1.0,
          target_layers: [67],
          normalize: true,
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
