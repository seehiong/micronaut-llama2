package example.micronaut.utils;

import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;

import example.micronaut.model.Tokenizer;
import io.micronaut.core.util.StringUtils;
import lombok.experimental.UtilityClass;

@UtilityClass
public class TokenUtils {

    // ----------------------------------------------------------------------------
    // The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens
    public String decode(Tokenizer t, int prev_token, int token) {
        String piece = t.vocab[token];

        // following BOS (1) token, sentencepiece decoder strips any leading whitespace
        // (see PR #89)
        if (prev_token == 1 && piece.charAt(0) == ' ') {
            piece = piece.substring(1);
        }

        // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
        String prefix = "<0x";
        String suffix = ">";
        if (piece.length() == 6 && piece.startsWith(prefix) && piece.endsWith(suffix)) {
            String hex2 = piece.substring(prefix.length(), prefix.length() + 2);
            char ch = (char) Integer.parseInt(hex2, 16);

            // ok this token is a raw byte token, carefuly to only print printable chars or
            // whitespace
            // some of the other bytes can be various control codes, backspace, etc. => skip
            piece = Character.toString(ch);
        }
        return piece;
    }

    public String safe_printf(String piece) {
        // piece might be a raw byte token, and we only want to print printable chars or
        // whitespace
        // because some of the other bytes can be various control codes, backspace, etc.
        if (piece == null) {
            return StringUtils.EMPTY_STRING;
        }
        if (piece.isEmpty()) {
            return StringUtils.EMPTY_STRING;
        }
        if (piece.length() == 1) {
            char ch = piece.charAt(0);
            boolean isPrintable = (32 <= ch && ch < 127);
            if (!(isPrintable || Character.isWhitespace(ch))) {
                return StringUtils.EMPTY_STRING;
            }
        }
        return piece;
    }

    public int str_lookup(String str, Map<String, Integer> sorted_vocab) {
        // efficiently find the perfect match for str in vocab, return its index or -1
        // if not found
        return sorted_vocab.getOrDefault(str, -1);
    }

    public int encode(Tokenizer t, String text, boolean bos, boolean eos, int[] tokens) {
        // encode the string text (input) into an upper-bound preallocated tokens[]
        // array
        // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS
        // token (=2)
        if (text == null) {
            System.err.println("cannot encode NULL text");
            System.exit(1);
        }

        if (t.sorted_vocab == null) {
            // sort vocabulary
            t.sorted_vocab = new HashMap<>();
            for (int i = 0; i < t.vocab_size; i++) {
                assert !t.sorted_vocab.containsKey(t.vocab[i]);
                t.sorted_vocab.put(t.vocab[i], i);
            }
        }

        // start at 0 tokens
        int n_tokens = 0; // the number of tokens

        // add optional BOS (=1) token, if desired
        if (bos) {
            tokens[n_tokens++] = 1;
        }

        // so prepend a dummy prefix token to the input string, but only if text != ""
        // TODO: pretty sure this isn't correct in the general case but I don't have the
        // energy to read more of the sentencepiece code to figure out what it's doing
        if (!"".equals(text)) {
            int dummy_prefix = str_lookup(" ", t.sorted_vocab);
            tokens[n_tokens++] = dummy_prefix;
        }

        // first encode every individual codepoint in the input string
        for (int i = 0, cpi; i < text.length(); i += Character.charCount(cpi)) {
            cpi = text.codePointAt(i);

            String singleCodepoint = Character.toString(cpi);
            int id = str_lookup(singleCodepoint, t.sorted_vocab);

            if (id != -1) {
                // we found this codepoint in vocab, add it as a token
                tokens[n_tokens++] = id;
            } else {
                // byte_fallback encoding: just encode each byte as a token
                // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
                // so the individual bytes only start at index 3
                for (byte b : singleCodepoint.getBytes(StandardCharsets.UTF_8)) {
                    tokens[n_tokens++] = Byte.toUnsignedInt(b) + 3;
                }
            }
        }

        // merge the best consecutive pair each iteration, according the scores in
        // vocab_scores
        while (true) {
            float best_score = -1e10f;
            int best_id = -1;
            int best_idx = -1;

            for (int i = 0; i < n_tokens - 1; ++i) {
                // check if we can merge the pair (tokens[i], tokens[i+1])
                String str_buffer = t.vocab[tokens[i]] + t.vocab[tokens[i + 1]];
                int id = str_lookup(str_buffer, t.sorted_vocab);
                if (id != -1 && t.vocab_scores[id] > best_score) {
                    // this merge pair exists in vocab! record its score and position
                    best_score = t.vocab_scores[id];
                    best_id = id;
                    best_idx = i;
                }
            }

            if (best_idx == -1) {
                break; // we couldn't find any more pairs to merge, so we're done
            }

            // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
            tokens[best_idx] = best_id;
            // delete token at position best_idx+1, shift the entire sequence back 1
            for (int i = best_idx + 1; i < n_tokens - 1; i++) {
                tokens[i] = tokens[i + 1];
            }
            n_tokens--; // token length decreased
        }

        // add optional EOS (=2) token, if desired
        if (eos) {
            tokens[n_tokens++] = 2;
        }

        return n_tokens;
    }
}
