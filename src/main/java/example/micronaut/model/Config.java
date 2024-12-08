package example.micronaut.model;

import java.nio.ByteBuffer;

import lombok.ToString;

@ToString
public class Config {

    public final int dim; // transformer dimension
    public final int hidden_dim; // for ffn layers
    public final int n_layers; // number of layers
    public final int n_heads; // number of query heads
    public final int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    public final int vocab_size; // vocabulary size, usually 256 (byte-level)
    public final int seq_len; // max sequence length
    public final boolean shared_weights;
    public final int head_size;

    Config(ByteBuffer buffer) {
        this.dim = buffer.getInt();
        this.hidden_dim = buffer.getInt();
        this.n_layers = buffer.getInt();
        this.n_heads = buffer.getInt();
        this.n_kv_heads = buffer.getInt();
        int vocab_size = buffer.getInt();
        this.vocab_size = Math.abs(vocab_size);
        this.seq_len = buffer.getInt();
        this.shared_weights = vocab_size > 0;
        this.head_size = dim / n_heads;
    }
}
