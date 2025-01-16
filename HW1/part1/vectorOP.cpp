#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

// Helper function to clamp the result to a maximum threshold
inline void clampResult(__pp_vec_float &result, __pp_vec_float &threshold, __pp_mask &validMask) {
  __pp_mask overflowMask = _pp_init_ones(0);
  _pp_vlt_float(overflowMask, threshold, result, validMask);  // Check if result exceeds the threshold
  _pp_vmove_float(result, threshold, overflowMask);           // Clamp values to the threshold
}

void clampedExpVector(float *inputValues, int *inputExponents, float *outputValues, int length)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //
  const float maxThreshold = 9.999999f;
  int remainingElements = length;
  
  __pp_vec_float currentResult, baseValues, threshold = _pp_vset_float(maxThreshold);
  __pp_vec_int remainingExp;
  __pp_vec_int zeroInt = _pp_vset_int(0), oneInt = _pp_vset_int(1);
  __pp_mask validMask, expMask;

  for (int i = 0; i < length; i += VECTOR_WIDTH) {
    // Initialize valid mask based on remaining elements
    validMask = _pp_init_ones(remainingElements);
    remainingElements -= VECTOR_WIDTH;

    // Load base values and exponents
    _pp_vload_int(remainingExp, inputExponents + i, validMask);
    _pp_vload_float(baseValues, inputValues + i, validMask);

    // Initialization current result
    _pp_vset_float(currentResult, 1.0f, validMask);

    // where exp > 0 
    expMask = _pp_init_ones(0);
    _pp_vlt_int(expMask, zeroInt, remainingExp, validMask);

    // Loop until all exponents are reduced to 0
    while (_pp_cntbits(expMask)) {
      _pp_vmult_float(currentResult, currentResult, baseValues, expMask); // Multiply by base values
      
      // Clamp results that exceed the threshold
      clampResult(currentResult, threshold, validMask);
      
      // exponents[i] -= 1
      _pp_vsub_int(remainingExp, remainingExp, oneInt, expMask);

      // Update expMask (where exp > 0)
      _pp_vlt_int(expMask, zeroInt, remainingExp, validMask);
    }

    // Store the final results
    _pp_vstore_float(outputValues + i, currentResult, validMask);
  }
}

inline float reduceVectorToSum(__pp_vec_float &vector) {
  int vecWidth = VECTOR_WIDTH;
  __pp_vec_float result;
  __pp_mask activeMask = _pp_init_ones();

  while (vecWidth > 1) {
    _pp_hadd_float(vector, vector);                   // Horizontal addition
    _pp_interleave_float(result, vector);             // Interleave to handle even-odd elements
    _pp_vmove_float(vector, result, activeMask);      // Copy result back to vector
    vecWidth /= 2;                                    
  }

  return vector.value[0]; 
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *inputArray, int length)
{

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //
  float totalSum = 0.0f;

  __pp_vec_float partialSum;
  __pp_mask activeMask = _pp_init_ones();

  for (int i = 0; i < length; i += VECTOR_WIDTH) {
      _pp_vload_float(partialSum, inputArray + i, activeMask);  // Load vector from input array

      // Reduce the vector to a scalar sum
      totalSum += reduceVectorToSum(partialSum);
  }

  return totalSum;
}