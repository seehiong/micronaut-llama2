package example.micronaut.service;

import example.micronaut.model.Sampler;
import example.micronaut.model.Tokenizer;
import example.micronaut.model.Transformer;
import example.micronaut.utils.TransformerUtils;
import example.micronaut.utils.SamplingUtils;
import example.micronaut.utils.TokenUtils;

import jakarta.inject.Singleton;

@Singleton
public class Llama2Service {

    // ----------------------------------------------------------------------------
    // utilities: time / rng
    public long time_in_ms() {
        // return time in milliseconds, for benchmarking the model speed
        return System.nanoTime() / 1_000_000;
    }

    // ----------------------------------------------------------------------------
    // generation loop
    public String generate(Transformer transformer, Tokenizer tokenizer, Sampler sampler, String prompt, int steps) {
        StringBuffer sb = new StringBuffer();

        String empty_prompt = "";
        if (prompt == null) {
            prompt = empty_prompt;
        }

        // encode the (string) prompt into tokens sequence
        int num_prompt_tokens = 0; // the total number of prompt tokens
        int[] prompt_tokens = new int[prompt.length() * 2 + 3]; // +3 for '\0', ?BOS, ?EOS
        num_prompt_tokens = TokenUtils.encode(tokenizer, prompt, true, false, prompt_tokens);
        if (num_prompt_tokens < 1) {
            sb.append("something is wrong, expected at least 1 prompt token");
            return sb.toString();
        }

        // start the main loop
        long start = 0; // used to time our code, only initialized after first iteration
        int next; // will store the next token in the sequence
        int token = prompt_tokens[0]; // kick off with the first token in the prompt
        int pos = 0; // position in the sequence
        while (pos < steps) {
            // forward the transformer to get logits for the next token
            float[] logits = TransformerUtils.forward(transformer, token, pos);

            // advance the state machine
            if (pos < num_prompt_tokens - 1) {
                // if we are still processing the input prompt, force the next prompt token
                next = prompt_tokens[pos + 1];
            } else {
                // otherwise sample the next token from the logits
                next = SamplingUtils.sample(sampler, logits);
            }
            pos++;

            // data-dependent terminating condition: the BOS (=1) token delimits sequences
            if (next == 1) {
                break;
            }

            // print the token as string, decode it with the Tokenizer object
            String piece = TokenUtils.decode(tokenizer, token, next);
            sb.append(TokenUtils.safe_printf(piece));

            token = next;

            // init the timer here because the first iteration can be slower
            if (start == 0) {
                start = time_in_ms();
            }
        }

        sb.append("\n");

        // report achieved tok/s (pos-1 because the timer starts after first iteration)
        if (pos > 1) {
            long end = time_in_ms();
            double tokensPerSecond = (pos - 1) / (double) (end - start) * 1000;
            sb.append("\nachieved tok/s: ").append(tokensPerSecond).append("\n");
        }

        return sb.toString();
    }

    // ----------------------------------------------------------------------------
    // chat loop
    // I manually inspected the tokens for a few chat conversations compared to
    // python reference and that seemed ok, but this was not thoroughly tested and
    // is not safely implemented, it's more a proof of concept atm.
    public String chat(Transformer transformer, Tokenizer tokenizer, Sampler sampler,
            String cli_user_prompt, String cli_system_prompt, int steps) {
        StringBuffer sb = new StringBuffer();

        // buffers for reading the system prompt and user prompt from stdin
        String system_prompt = null;
        String user_prompt = null;
        String rendered_prompt = null;
        int num_prompt_tokens = 0;
        int[] prompt_tokens = new int[512];
        int user_idx = 0;

        // start the main loop
        boolean user_turn = true; // user starts
        int next = 0; // will store the next token in the sequence
        int token = 0; // stores the current token to feed into the transformer
        int pos = 0; // position in the sequence
        while (pos < steps) {

            // when it is the user's turn to contribute tokens to the dialog...
            if (user_turn) {
                // get the (optional) system prompt at position 0
                if (pos == 0) {
                    // at position 0, the user can also contribute a system prompt
                    if (cli_system_prompt == null) {
                        // system prompt was not passed in, attempt to get it from stdin
                        system_prompt = SamplingUtils.read_stdin("Enter system prompt (optional): ");
                    } else {
                        // system prompt was passed in, use it
                        system_prompt = cli_system_prompt;
                    }
                }
                // get the user prompt
                if (pos == 0 && cli_user_prompt != null) {
                    // user prompt for position 0 was passed in, use it
                    user_prompt = cli_user_prompt;
                } else {
                    // otherwise get user prompt from stdin
                    user_prompt = SamplingUtils.read_stdin("User: ");
                }
                // render user/system prompts into the Llama 2 Chat schema
                if (pos == 0 && system_prompt.isEmpty()) {
                    String system_template = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
                    rendered_prompt = system_template.formatted(system_prompt, user_prompt);
                } else {
                    String user_template = "[INST] %s [/INST]";
                    rendered_prompt = user_template.formatted(user_prompt);
                }
                // encode the rendered prompt into tokens
                num_prompt_tokens = TokenUtils.encode(tokenizer, rendered_prompt, true, false, prompt_tokens);
                user_idx = 0; // reset the user index
                user_turn = false;
                sb.append("Assistant: ");
            }

            // determine the token to pass into the transformer next
            if (user_idx < num_prompt_tokens) {
                // if we are still processing the input prompt, force the next prompt token
                token = prompt_tokens[user_idx++];
            } else {
                // otherwise use the next token sampled from previous turn
                token = next;
            }
            // EOS (=2) token ends the Assistant turn
            if (token == 2) {
                user_turn = true;
            }

            // forward the transformer to get logits for the next token
            float[] logits = TransformerUtils.forward(transformer, token, pos);
            next = SamplingUtils.sample(sampler, logits);
            pos++;

            if (user_idx >= num_prompt_tokens && next != 2) {
                // the Assistant is responding, so print its output
                String piece = TokenUtils.decode(tokenizer, token, next);
                sb.append(TokenUtils.safe_printf(piece)); // same as printf("%s", piece), but skips "unsafe"
                // bytes
            }
            if (next == 2) {
                sb.append("\n");
            }
        }
        return sb.toString();
    }

}
