function tests = test_pilot_roundoff
tests = functiontests(localfunctions);
end

function testTinyFftResidueIsZeroed(testCase)
X = single([0, 0.0054, -2.4557e-12]);
[actual, info] = pilot_sanitize_roundoff(X, 'tiny residue');
verifyEqual(testCase, actual, single([0, 0.0054, 0]));
verifyEqual(testCase, info.negative_count_before_roundoff_cleanup, 1);
verifyEqual(testCase, info.roundoff_clipped_count, 1);
verifyGreaterThan(testCase, info.roundoff_tolerance_abs, abs(double(X(3))));
end

function testPeakRelativeLimitRejectsMaterialOutlier(testCase)
X = single([ones(1,10000), -3e-3]);
verifyError(testCase, @() pilot_sanitize_roundoff(X, 'material peak outlier'), ...
    'pilot:MaterialNegativeReconstruction');
end

function testAccumulationBoundResidueIsZeroedWhenMassIsNegligible(testCase)
X = single([ones(1,2000), -1.5e-3]);
[actual, info] = pilot_sanitize_roundoff(X, 'accumulation-bound residue');
verifyEqual(testCase, actual(end), single(0));
verifyEqual(testCase, actual(1:end-1), X(1:end-1));
verifyLessThan(testCase, info.minimum_relative_to_peak, ...
    info.roundoff_relative_peak_limit);
verifyLessThan(testCase, info.negative_to_positive_mass_ratio, ...
    info.roundoff_negative_mass_ratio_limit);
verifyEqual(testCase, info.roundoff_accumulation_terms, 24010);
end

function testObservedSubTwoPpmMassIsNumericallyNegligible(testCase)
% Mirrors the T01 Taylor failure geometry: local peak gate passes and the
% total clipped mass is 1.25 ppm, well below half a uint16 level.
X=single([ones(1,1000),-1.25e-3]);
[actual,info]=pilot_sanitize_roundoff(X,'sub-two-ppm residue');
verifyEqual(testCase,actual(end),single(0));
verifyEqual(testCase,info.negative_to_positive_mass_ratio,1.25e-6,'RelTol',1e-6);
verifyEqual(testCase,info.roundoff_negative_mass_ratio_limit,2e-6);
end

function testAboveTwoPpmMassStillFailsWhenPeakGatePasses(testCase)
X=single([ones(1,1000),-1.5e-3,-1.5e-3]);
verifyError(testCase,@() pilot_sanitize_roundoff(X,'above-two-ppm residue'), ...
    'pilot:MaterialNegativeReconstruction');
end

function testNegativeMassLimitRejectsDistributedContent(testCase)
X = single([ones(1,10), -8e-5*ones(1,1000)]);
verifyError(testCase, @() pilot_sanitize_roundoff(X, 'material negative mass'), ...
    'pilot:MaterialNegativeReconstruction');
end

function testNonnegativeInputIsBitExact(testCase)
X = single([0, realmin('single'), 0.25, 1]);
[actual, info] = pilot_sanitize_roundoff(X, 'nonnegative');
verifyEqual(testCase, actual, X);
verifyEqual(testCase, info.negative_count_before_roundoff_cleanup, 0);
verifyEqual(testCase, info.roundoff_clipped_count, 0);
end
