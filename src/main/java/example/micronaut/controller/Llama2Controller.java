package example.micronaut.controller;

import java.io.IOException;

import example.micronaut.model.Sampler;
import example.micronaut.model.Tokenizer;
import example.micronaut.model.Transformer;
import example.micronaut.service.Llama2Service;
import io.micronaut.context.annotation.Value;
import io.micronaut.http.annotation.*;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;

@Controller("/api/llama2")
@RequiredArgsConstructor
public class Llama2Controller {

    private final Llama2Service llama2Service;
    private int steps = 256; // max number of steps to run for, 0: use seq_len

    private Transformer transformer;
    private Tokenizer tokenizer;
    private Sampler sampler;

    @Value("${transformer.checkpoint_path}")
    private String checkpoint_path;
    @Value("${transformer.tokenizer_path}")
    private String tokenizer_path;

    @PostConstruct
    public void init() throws IOException {
        // default parameters
        float temperature = 1.0f; // 0.0 = greedy deterministic. 1.0 = original. don't set higher
        float topp = 0.9f; // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
        long rng_seed = 0; // seed rng with time by default

        // parameter validation/overrides
        if (rng_seed <= 0) {
            rng_seed = System.currentTimeMillis();
        }
        if (temperature < 0.0) {
            temperature = 0.0f;
        }
        if (topp < 0.0 || 1.0 < topp) {
            topp = 0.9f;
        }
        if (steps <= 0) {
            steps = 0;
        }

        // build the Transformer via the model .bin file
        transformer = new Transformer(checkpoint_path);
        if (steps == 0 || steps > transformer.config.seq_len) {
            steps = transformer.config.seq_len; // ovrerride to ~max length
        }

        // build the Tokenizer via the tokenizer .bin file
        tokenizer = new Tokenizer(tokenizer_path, transformer.config.vocab_size);

        // build the Sampler
        sampler = new Sampler(transformer.config.vocab_size, temperature, topp, rng_seed);
    }

    @Get("/generate")
    public String generate(@QueryValue(defaultValue = "Once upon a time") String prompt) {
        return llama2Service.generate(transformer, tokenizer, sampler, prompt, steps);
    }

    @Get("/chat")
    public String chat(@QueryValue(defaultValue = "Once upon a time") String prompt,
            @QueryValue(defaultValue = "You are a helpful assistant.") String system_prompt) {
        return llama2Service.chat(transformer, tokenizer, sampler, prompt, system_prompt, steps);
    }
}
