// The tensor element types a GGUF file can hold: ggml's types, by the
// numbers ggml gives them. Block and type sizes are ggml's own
// (ggml_blck_size, ggml_type_size), checked against llama.cpp.
package gguf

/// TensorType is how a tensor's elements are stored: a plain scalar, or a
/// quantized block of BlockSize elements in TypeSize bytes.
public enum TensorType: uint32 {
    case F32 = 0
    case F16 = 1
    case Q4_0 = 2
    case Q4_1 = 3
    case Q5_0 = 6
    case Q5_1 = 7
    case Q8_0 = 8
    case Q8_1 = 9
    case Q2_K = 10
    case Q3_K = 11
    case Q4_K = 12
    case Q5_K = 13
    case Q6_K = 14
    case Q8_K = 15
    case IQ2_XXS = 16
    case IQ2_XS = 17
    case IQ3_XXS = 18
    case IQ1_S = 19
    case IQ4_NL = 20
    case IQ3_S = 21
    case IQ2_S = 22
    case IQ4_XS = 23
    case I8 = 24
    case I16 = 25
    case I32 = 26
    case I64 = 27
    case F64 = 28
    case IQ1_M = 29
    case BF16 = 30
    case TQ1_0 = 34
    case TQ2_0 = 35
    case MXFP4 = 39
    case NVFP4 = 40
    case Q1_0 = 41
    case Q2_0 = 42

    /// Name is ggml's name for the type: "f32", "q4_0", "q4_K".
    public var Name: string {
        switch self {
        case .F32: return "f32"
        case .F16: return "f16"
        case .Q4_0: return "q4_0"
        case .Q4_1: return "q4_1"
        case .Q5_0: return "q5_0"
        case .Q5_1: return "q5_1"
        case .Q8_0: return "q8_0"
        case .Q8_1: return "q8_1"
        case .Q2_K: return "q2_K"
        case .Q3_K: return "q3_K"
        case .Q4_K: return "q4_K"
        case .Q5_K: return "q5_K"
        case .Q6_K: return "q6_K"
        case .Q8_K: return "q8_K"
        case .IQ2_XXS: return "iq2_xxs"
        case .IQ2_XS: return "iq2_xs"
        case .IQ3_XXS: return "iq3_xxs"
        case .IQ1_S: return "iq1_s"
        case .IQ4_NL: return "iq4_nl"
        case .IQ3_S: return "iq3_s"
        case .IQ2_S: return "iq2_s"
        case .IQ4_XS: return "iq4_xs"
        case .I8: return "i8"
        case .I16: return "i16"
        case .I32: return "i32"
        case .I64: return "i64"
        case .F64: return "f64"
        case .IQ1_M: return "iq1_m"
        case .BF16: return "bf16"
        case .TQ1_0: return "tq1_0"
        case .TQ2_0: return "tq2_0"
        case .MXFP4: return "mxfp4"
        case .NVFP4: return "nvfp4"
        case .Q1_0: return "q1_0"
        case .Q2_0: return "q2_0"
        }
    }

    /// BlockSize is how many elements one block holds: 1 for a scalar.
    public var BlockSize: int {
        switch self {
        case .F32, .F16, .I8, .I16, .I32, .I64, .F64, .BF16: return 1
        case .Q4_0, .Q4_1, .Q5_0, .Q5_1, .Q8_0, .Q8_1, .IQ4_NL, .MXFP4: return 32
        case .NVFP4, .Q2_0: return 64
        case .Q1_0: return 128
        case .Q2_K, .Q3_K, .Q4_K, .Q5_K, .Q6_K, .Q8_K, .IQ2_XXS, .IQ2_XS, .IQ3_XXS, .IQ1_S, .IQ3_S, .IQ2_S, .IQ4_XS, .IQ1_M, .TQ1_0, .TQ2_0: return 256
        }
    }

    /// TypeSize is how many bytes one block takes.
    public var TypeSize: int {
        switch self {
        case .I8: return 1
        case .F16, .I16, .BF16: return 2
        case .F32, .I32: return 4
        case .I64, .F64: return 8
        case .MXFP4: return 17
        case .Q4_0, .IQ4_NL, .Q1_0, .Q2_0: return 18
        case .Q4_1: return 20
        case .Q5_0: return 22
        case .Q5_1: return 24
        case .Q8_0: return 34
        case .Q8_1, .NVFP4: return 36
        case .IQ1_S: return 50
        case .TQ1_0: return 54
        case .IQ1_M: return 56
        case .IQ2_XXS, .TQ2_0: return 66
        case .IQ2_XS: return 74
        case .IQ2_S: return 82
        case .Q2_K: return 84
        case .IQ3_XXS: return 98
        case .Q3_K, .IQ3_S: return 110
        case .IQ4_XS: return 136
        case .Q4_K: return 144
        case .Q5_K: return 176
        case .Q6_K: return 210
        case .Q8_K: return 292
        }
    }

    /// IsQuantized is whether elements are stored in blocks with a scale.
    public var IsQuantized: bool {
        switch self {
        case .F32, .F16, .BF16, .F64, .I8, .I16, .I32, .I64: return false
        default: return true
        }
    }

    /// RowSize is the bytes a row of n elements takes; n is a multiple of
    /// BlockSize.
    public func RowSize(_ n: int) -> int {
        return n / BlockSize * TypeSize
    }
}

/// ValueType is the type of a metadata value, by GGUF's numbers.
public enum ValueType: uint32 {
    case U8 = 0
    case I8 = 1
    case U16 = 2
    case I16 = 3
    case U32 = 4
    case I32 = 5
    case F32 = 6
    case Bool = 7
    case Text = 8
    case Array = 9
    case U64 = 10
    case I64 = 11
    case F64 = 12
}
