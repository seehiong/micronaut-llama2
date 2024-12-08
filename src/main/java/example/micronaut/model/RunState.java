package example.micronaut.model;

import lombok.Getter;

@Getter
public class RunState {

    // current wave of activations
    public final float[] x; // activation at current time stamp (dim,)
    public final float[] xb; // same, but inside a residual branch (dim,)
    public final float[] xb2; // an additional buffer just for convenience (dim,)
    public final float[] hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    public final float[] hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    public final float[] q; // query (dim,)
    public final float[] k; // key (dim,)
    public final float[] v; // value (dim,)
    public final float[] att; // buffer for scores/attention values (n_heads, seq_len)
    public final float[] logits; // output logits

    // kv cache
    public final float[][] key_cache; // (layer, seq_len, dim)
    public final float[][] value_cache; // (layer, seq_len, dim)

    RunState(Config config) {
        int kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
        this.x = new float[config.dim];
        this.xb = new float[config.dim];
        this.xb2 = new float[config.dim];
        this.hb = new float[config.hidden_dim];
        this.hb2 = new float[config.hidden_dim];
        this.q = new float[config.dim];
        this.k = new float[kv_dim];
        this.v = new float[kv_dim];
        this.att = new float[config.n_heads * config.seq_len];
        this.logits = new float[config.vocab_size];
        this.key_cache = new float[config.n_layers][config.seq_len * kv_dim];
        this.value_cache = new float[config.n_layers][config.seq_len * kv_dim];
    }
}
