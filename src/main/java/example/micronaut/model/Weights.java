package example.micronaut.model;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

import lombok.Getter;

@Getter
public class Weights {

    // token embedding table
    public final FloatBuffer token_embedding_table; // (vocab_size, dim)
    // weights for rmsnorms
    public final FloatBuffer[] rms_att_weight; // (layer, dim) rmsnorm weights
    // weights for matmuls. note dim == n_heads * head_size
    public final FloatBuffer[] wq; // (layer, dim, n_heads * head_size)
    public final FloatBuffer[] wk; // (layer, dim, n_kv_heads * head_size)
    public final FloatBuffer[] wv; // (layer, dim, n_kv_heads * head_size)
    public final FloatBuffer[] wo; // (layer, n_heads * head_size, dim)
    public final FloatBuffer[] rms_ffn_weight; // (layer, dim)
    // weights for ffn
    public final FloatBuffer[] w1; // (layer, hidden_dim, dim)
    public final FloatBuffer[] w2; // (layer, dim, hidden_dim)
    public final FloatBuffer[] w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    public final FloatBuffer rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    public final FloatBuffer wcls; // (vocab_size, dim)

    static FloatBuffer takeFloats(MemorySegment memorySegment, long[] position, int... dims) {
        long totalBytes = 1;
        for (int d : dims) {
            totalBytes *= d;
        }
        totalBytes *= Float.BYTES;
        MemorySegment slice = memorySegment.asSlice(position[0], totalBytes);
        position[0] += totalBytes;
        return slice.asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
    }

    static FloatBuffer[] takeArray(MemorySegment memorySegment, long[] position, int dim0, int... dims) {
        FloatBuffer[] segments = new FloatBuffer[dim0];
        for (int i = 0; i < dim0; ++i) {
            segments[i] = takeFloats(memorySegment, position, dims);
        }
        return segments;
    }

    // ----------------------------------------------------------------------------
    // initialization: read from checkpoint
    // read weights from memory segment
    Weights(Config config, MemorySegment memorySegment) {
        long[] position = new long[]{0};
        this.token_embedding_table = takeFloats(memorySegment, position, config.vocab_size, config.dim);
        this.rms_att_weight = takeArray(memorySegment, position, config.n_layers, config.dim);
        this.wq = takeArray(memorySegment, position, config.n_layers, config.dim, config.n_heads * config.head_size);
        this.wk = takeArray(memorySegment, position, config.n_layers, config.dim, config.n_kv_heads * config.head_size);
        this.wv = takeArray(memorySegment, position, config.n_layers, config.dim, config.n_kv_heads * config.head_size);
        this.wo = takeArray(memorySegment, position, config.n_layers, config.n_heads * config.head_size, config.dim);
        this.rms_ffn_weight = takeArray(memorySegment, position, config.n_layers, config.dim);
        this.w1 = takeArray(memorySegment, position, config.n_layers, config.hidden_dim, config.dim);
        this.w2 = takeArray(memorySegment, position, config.n_layers, config.dim, config.hidden_dim);
        this.w3 = takeArray(memorySegment, position, config.n_layers, config.hidden_dim, config.dim);
        this.rms_final_weight = takeFloats(memorySegment, position, config.dim);

        // skip what used to be freq_cis_real (for RoPE)
        position[0] += (config.seq_len * config.head_size / 2) * Float.BYTES;

        // skip what used to be freq_cis_imag (for RoPE)
        position[0] += (config.seq_len * config.head_size / 2) * Float.BYTES;

        this.wcls = config.shared_weights
                ? this.token_embedding_table
                : takeFloats(memorySegment, position, config.vocab_size, config.dim);
    }
}
