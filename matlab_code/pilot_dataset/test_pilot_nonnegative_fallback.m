function tests=test_pilot_nonnegative_fallback
tests=functiontests(localfunctions);
end

function testZeroSupportCannotCreateNanOrInf(testCase)
forward=@(x) x;
backward=@(x) x;
Htf=single([0,1,0;2,0,3]);
actual=pilot_deconv_rl_nonnegative(forward,backward,Htf,3);
verifyEqual(testCase,actual,Htf);
verifyTrue(testCase,all(isfinite(actual),'all'));
verifyGreaterThanOrEqual(testCase,min(actual,[],'all'),single(0));
end

function testUnsupportedPositiveNumeratorIsSetToZero(testCase)
forward=@(x) zeros(size(x),'like',x);
backward=@(x) zeros(size(x),'like',x);
Htf=single([1,0;0,2]);
actual=pilot_deconv_rl_nonnegative(forward,backward,Htf,1);
verifyEqual(testCase,actual,zeros(size(Htf),'single'));
verifyTrue(testCase,all(isfinite(actual),'all'));
end

function testNegativeOperatorResidueIsOutsidePhysicalCone(testCase)
forward=@(x) x;
backward=@(x) x-single(2);
Htf=single([1,3]);
actual=pilot_deconv_rl_nonnegative(forward,backward,Htf,2);
verifyTrue(testCase,all(isfinite(actual),'all'));
verifyGreaterThanOrEqual(testCase,min(actual,[],'all'),single(0));
end
