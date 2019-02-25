import numpy
import math

def binornd(X):
    P = numpy.zeros(X.shape, dtype=numpy.float32)
    P[X>numpy.float32(0.5)] = numpy.float32(1.)
    return P

def rounding(W, integer_width, float_width):
    power = numpy.float32(2**float_width)
    integer_data = numpy.float32(numpy.floor(W * power, dtype=numpy.float32))
    float_data = numpy.float32(integer_data / power)
    P = numpy.float32((W - float_data) * power)
    return numpy.float32(binornd(P) * (1/power) + float_data)

def fixed_point_quantization(W, integer_width, float_width):
    minValue = numpy.float32(-1 * (2**(integer_width-1)))
    maxValue = numpy.float32(2**(integer_width-1) - 2**(-float_width))

    Q = numpy.copy(W)
    Q[Q>=maxValue] = maxValue
    Q[Q<=minValue] = minValue
    Q = rounding(Q, integer_width, float_width)
    return Q

def power_quantization(W, integer_width, float_width, kernelwise):
    W = W.astype(numpy.float32)
    sign = numpy.ones(W.shape, dtype=numpy.float32)
    sign[W<0] = numpy.float32(-1.)

    Q = numpy.ones(W.shape, dtype=numpy.float32)
    W_abs = numpy.absolute(W, dtype=numpy.float32)

    bitwidth = integer_width+float_width
    exponent = numpy.power(2, bitwidth-1, dtype=numpy.float32)-2 # 6 in 4-bit
    
    exp_min = integer_width
    exp_max = integer_width+exponent

    abs_min = numpy.power(2, exp_min, dtype=numpy.float32) # 2^min
    abs_max = numpy.power(2, exp_max, dtype=numpy.float32) # 2^(min+6)

    exp = numpy.log2(W_abs, dtype=numpy.float32)
    lower_bound = numpy.floor(exp)
    upper_bound = lower_bound+1
    power_lower_bound = numpy.power(2, lower_bound, dtype=numpy.float32)
    power_upper_bound = numpy.power(2, upper_bound, dtype=numpy.float32)
    # power_min_bound = numpy.minimum(power_lower_bound, power_upper_bound)
    # power_max_bound = numpy.maximum(power_lower_bound, power_upper_bound)
    threshold = power_lower_bound + power_upper_bound
    threshold /= 2

    Q[W_abs<=threshold] = power_lower_bound[W_abs<=threshold]
    Q[W_abs>threshold] = power_upper_bound[W_abs>threshold]

    if kernelwise == True:
        ones = numpy.ones((1,W.shape[1]), dtype=numpy.float32)
        abs_min = numpy.dot(abs_min, ones)
        abs_max = numpy.dot(abs_max, ones)
        Q[Q<=abs_min/2] = numpy.float32(0.)
        Q[Q>=abs_max] = abs_max[Q>=abs_max]
    else:
        Q[Q<=abs_min/2] = numpy.float32(0.)
        Q[Q>=abs_max] = abs_max

    Q = sign*Q
    return Q

def weight_quantization(W, integer_width, float_width, qtype="power", kernelwise=False):
    if qtype=="power":
        Q = power_quantization(W, integer_width, float_width, kernelwise)
    else:
        assert qtype=="fixed", "Unknown quantization type!"
        Q = fixed_point_quantization(W, integer_width, float_width)
    return Q

def alpha_init(W):
    kernel_dim = W.shape[1]
    ones = numpy.ones((kernel_dim,1), dtype=numpy.float32)
    return  numpy.dot(W*W, ones) / numpy.float32(kernel_dim)

def argmin_B(W, alpha, integer_width, float_width, qtype="power", kernelwise=False):
    B = W / numpy.float32(alpha)
    B = weight_quantization(B, integer_width, float_width, qtype, kernelwise)
    return B

def argmin_alpha(W, Q):
    kernel_dim = W.shape[1]
    ones = numpy.ones((kernel_dim,1), dtype=numpy.float32)
    alpha = numpy.dot(W*Q, ones) / (numpy.dot(Q*Q, ones)+1e-7)
    return alpha

def kernel_mse(W, Q):
    kernel_dim = W.shape[1]
    ones = numpy.ones((kernel_dim,1), dtype=numpy.float32)
    Diff = W-Q
    return  numpy.dot(Diff*Diff, ones)

def quantization_with_scale(W, alpha_v, qtype="power"):
    W_shape = W.shape
    alpha_shape = alpha_v.shape
    W = W.astype('float32')
    try:
        n = W.shape[0]
        c = W.shape[1]
        h = W.shape[2]
        w = W.shape[3]
    except:
        n = W.shape[0]
        c = W.shape[1]
        h = 1
        w = 1
    W.resize(n,c*h*w)
    epsilon = 1e-7
    if qtype=="power":
        # minError = 1e10*numpy.ones((W.shape[0],1), dtype=numpy.float32)
        # best_width = numpy.zeros((W.shape[0],1), dtype=numpy.int32)
        minError = 1e10
        best_width = 0
        start_width = 0
        # N = [0,1,2,3]
        # 0,2,4,8
        for width in range(0, 4):
            prev_alpha = 1e10
            alpha = alpha_init(W)
            Q = argmin_B(W, alpha, start_width+width, 4-(start_width+width))
            diff = prev_alpha-alpha
            div = numpy.linalg.norm(diff,2)/numpy.linalg.norm(alpha,2)
            prev_alpha = alpha
            while div > numpy.float64(epsilon):
                alpha = argmin_alpha(W, Q)
                Q = argmin_B(W, alpha, start_width+width, 4-(start_width+width))
                diff = prev_alpha-alpha
                div = numpy.linalg.norm(diff,2)/numpy.linalg.norm(alpha,2)
                prev_alpha = alpha
            Q = Q*alpha
            # error = kernel_mse(W, Q)
            # best_width[error < minError] = width
            # minError[error < minError] = error[error < minError]
            # print "2^n Minimum \t: ", numpy.power(2.,start_width+width), "\t| LOSS : ", error.mean(), "\t| MIN LOSS ", minError.mean()
        # print "MIN LOSS ", minError.mean()
            error = numpy.dot(W.ravel()-Q.ravel(),W.ravel()-Q.ravel())
            if error < minError:
                minError = error
                best_width = width
            print "2^n starts from \t: ", numpy.power(2.,start_width+width), "\t| LOSS : ", error, "\t| MIN LOSS ", minError
        print "BEST 2^n starts from \t: ", numpy.power(2.,start_width+best_width), "\t| LOSS : ", minError, "\t| MIN LOSS ", minError
        prev_alpha = 1e10
        alpha = alpha_init(W)
        Q = argmin_B(W, alpha, start_width+best_width, 4-(start_width+best_width), kernelwise=False)
        diff = prev_alpha-alpha
        div = numpy.linalg.norm(diff,2)/numpy.linalg.norm(alpha,2)
        prev_alpha = alpha
        while div > numpy.float64(epsilon):
            alpha = argmin_alpha(W, Q)
            Q = argmin_B(W, alpha, start_width+best_width, 4-(start_width+best_width), kernelwise=False)
            diff = prev_alpha-alpha
            div = numpy.linalg.norm(diff,2)/numpy.linalg.norm(alpha,2)
            prev_alpha = alpha
        Q.resize(W_shape)
        alpha.resize(alpha_shape)
        return Q,alpha
    else:
        assert qtype=="fixed", "Unknown quantization type!"
        minError = 1e10
        best_width = 1
        for width in range(1, 9):
            prev_alpha = 1e10
            alpha = alpha_init(W)
            Q = argmin_B(W, alpha, width, 8-width, "fixed")
            diff = prev_alpha-alpha
            div = numpy.linalg.norm(diff,2)/numpy.linalg.norm(alpha,2)
            prev_alpha = alpha
            while div > numpy.float64(epsilon):
                alpha = argmin_alpha(W, Q)
                Q = argmin_B(W, alpha, width, 8-width, "fixed")
                diff = prev_alpha-alpha
                div = numpy.linalg.norm(diff,2)/numpy.linalg.norm(alpha,2)
                prev_alpha = alpha
            Q = Q*alpha
            error = numpy.dot(W.ravel()-Q.ravel(),W.ravel()-Q.ravel())
            if error < minError:
                minError = error
                best_width = width
            print "INT Bitwidth \t: ", width, "\t| LOSS : ", error, "\t| MIN LOSS ", minError
        print "BEST Bitwidth \t: ", best_width, "\t| LOSS : ", minError, "\t| MIN LOSS ", minError
        prev_alpha = 1e10
        alpha = alpha_init(W)
        Q = argmin_B(W, alpha, best_width, 8-best_width, "fixed")
        diff = prev_alpha-alpha
        div = numpy.linalg.norm(diff,2)/numpy.linalg.norm(alpha,2)
        prev_alpha = alpha
        while div > numpy.float64(epsilon):
            alpha = argmin_alpha(W, Q)
            Q = argmin_B(W, alpha, best_width, 8-best_width, "fixed")
            diff = prev_alpha-alpha
            div = numpy.linalg.norm(diff,2)/numpy.linalg.norm(alpha,2)
            prev_alpha = alpha
        Q.resize(W_shape)
        alpha.resize(alpha_shape)
        return Q,alpha
