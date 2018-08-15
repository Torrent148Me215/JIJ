############################################################
# Initial setup
############################################################

import matplotlib
import matplotlib.pyplot as plot
import numpy
import scipy.stats as stats
import multiprocessing
import math

import pystan
import stan_utility

help(stan_utility)

light="#DCBCBC"
light_highlight="#C79999"
mid="#B97C7C"
mid_highlight="#A25050"
dark="#8F2727"
dark_highlight="#7C0000"
green="#00FF00"

############################################################
#
# Workflow
#
############################################################

############################################################
# PRIOR TO OBSERVATION
############################################################

############################################################
# 1. Conceptual Analysis
############################################################

# We are working with a suite of detectors that each record
# the same source over a given time interval. The source
# strength and detector response are not expected to vary
# significantly in time.  Each detector is identical and
# returns discrete counts.

############################################################
# 2. Define Observations
############################################################

# Mathematically our observation takes the form of integer
# counts, y, for each of the N detectors.  In the Stan
# modeling lanugage this is specified as

with open('fit_data.stan', 'r') as file:
    lines = file.readlines()[0:4]
    for line in lines:
        print(line),

############################################################
# 3. Identify Relevant Summary Statistics
############################################################

# There are N components in each observation, one for each
# detector.  We could analyze each component independently,
# but because we assume that the detectors are all identical
# we can analyze their comprehensive responses with a histogram
# of their counts.  In other words we consider the histogram
# of detector counts _as the summary statistic_!

# In this conceptual example assume that our conceptual
# domain expertise informs us that 25 counts in a detector
# would be an extreme but not impossible observation.

############################################################
# 4. Build a Generative Model
############################################################

# The constant source strength and detector responds suggests
# a Poisson observation model for each of the detectors with
# a single source strength, lambda.
#
# Our domain expertise that 25 counts is extreme suggests that
# we want our prior for lambda to keep most of its probability
# mass below lambda = 15, which corresponds to fluctations in
# the observations around 15 + 3 * sqrt(15) ~ 25.
#
# We achieve this with a half-normal prior with standard
# deviation = 6.44787 such that only 1% of the prior probability
# mass is above lambda = 15.

ls = numpy.arange(0, 20, 0.001)
pdfs = [ stats.norm.pdf(l, 0, 6.44787) for l in ls]
plot.plot(ls, pdfs, linewidth=2, color=dark_highlight)

ls = numpy.arange(0, 15, 0.001)
pdfs = [ stats.norm.pdf(l, 0, 6.44787) for l in ls]
plot.fill_between(ls, 0, pdfs, color=dark_highlight)

plot.gca().set_xlabel("lambda")
plot.gca().set_ylabel("Prior Density")
plot.gca().axes.get_yaxis().set_visible(False)
plot.show()

# This generative model is implemented in the Stan programs
with open('generative_ensemble.stan', 'r') as file:
    print(file.read())

with open('fit_data.stan', 'r') as file:
    print(file.read())

############################################################
# 5. Analyze the Generative Ensemble
############################################################

R = 1000 # 1000 draws from the Bayesian joint distribution
N = 100

############################################################
# 5a. Analyze the Prior Predictive Distribution
############################################################

simu_data = dict(N = N)

model = stan_utility.compile_model('generative_ensemble.stan')
fit = model.sampling(data=simu_data,
                     iter=R, warmup=0, chains=1, refresh=R,
                     seed=4838282, algorithm="Fixed_param")

simu_lambdas = fit.extract()['lambda']
simu_ys = fit.extract()['y'].astype(numpy.int64)

# Plot aggregated summary histogram for simulated observations
max_y = 40
B = max_y + 1

bins = [b - 0.5 for b in range(B + 1)]

counts = [numpy.histogram(simu_ys[r], bins=bins)[0] for r in range(R)]
probs = [10, 20, 30, 40, 50, 60, 70, 80, 90]
creds = [numpy.percentile([count[b] for count in counts], probs)
         for b in range(B)]

idxs = [ idx for idx in range(B) for r in range(2) ]
xs = [ idx + delta for idx in range(B) for delta in [-0.5, 0.5]]
      
pad_creds = [ creds[idx] for idx in idxs ]

plot.fill_between(xs, [c[0] for c in pad_creds], [c[8] for c in pad_creds],
                  facecolor=light, color=light)
plot.fill_between(xs, [c[0] for c in pad_creds], [c[7] for c in pad_creds],
                  facecolor=light_highlight, color=light_highlight)
plot.fill_between(xs, [c[2] for c in pad_creds], [c[6] for c in pad_creds],
                  facecolor=mid, color=mid)
plot.fill_between(xs, [c[3] for c in pad_creds], [c[5] for c in pad_creds],
                  facecolor=mid_highlight, color=mid_highlight)
plot.plot(xs, [c[4] for c in pad_creds], color=dark)

plot.gca().set_xlim([min(bins), max(bins)])
plot.gca().set_xlabel("y")
plot.gca().set_ylim([0, max([c[8] for c in creds])])
plot.gca().set_ylabel("Prior Predictive Distribution")

plot.axvline(x=25, linewidth=2.5, color="white")
plot.axvline(x=25, linewidth=2, color="black")

plot.show()

# We see a very small prior predictive probability above the
# extreme observation scale from our domain expertise

float(len([ y for y in simu_ys.flatten() if y > 25 ])) / len(simu_ys.flatten())

############################################################
# 5b. Fit the simulated observations and evaluate
############################################################

simus = zip(simu_lambdas, simu_ys)
fit_model = stan_utility.compile_model('fit_data.stan')

def analyze_simu(simu):
    simu_l = simu[0]
    simu_y = simu[1]
    
    # Fit the simulated observation
    input_data = dict(N = N, y = simu_y)
    
    fit = fit_model.sampling(data=input_data, seed=4938483, n_jobs=1)
    
    # Compute diagnostics
    warning_code = stan_utility.check_all_diagnostics(fit, quiet=True)
    
    # Compute rank of prior draw with respect to thinned posterior draws
    thinned_l = fit.extract()['lambda'][numpy.arange(0, 4000 - 7, 8)]
    sbc_rank = len(filter(lambda x: x > simu_l, thinned_l))
    
    # Compute posterior sensitivities
    summary = fit.summary(probs=[0.5])
    post_mean_l = [x[0] for x in summary['summary']][0]
    post_sd_l = [x[2] for x in summary['summary']][0]
    
    prior_sd_l = 6.44787
    
    z_score = (post_mean_l - simu_l) / post_sd_l
    shrinkage = 1 - (post_sd_l / prior_sd_l)**2
    
    return [warning_code, sbc_rank, z_score, shrinkage]

pool = multiprocessing.Pool(4)
ensemble_output = pool.map(analyze_simu, simus)

# Check for fit diagnostics
warning_codes = [x[0] for x in ensemble_output]
if sum(warning_codes) is not 0:
    print ("Some posterior fits in the generative " +
           "ensemble encountered problems!")
    for r in range(R):
        if warning_codes[r] is not 0:
            print('Replication {} of {}'.format(r, R))
            print('Simulated lambda = {}'.format(simu_lambdas[r]))
            stan_utility.parse_warning_code(warning_codes[r])
            print("")
else:
    print ("No posterior fits in the generative " +
           "ensemble encountered problems!")

# Check SBC histogram
sbc_low = stats.binom.ppf(0.005, R, 25.0 / 500)
sbc_mid = stats.binom.ppf(0.5, R, 25.0 / 500)
sbc_high = stats.binom.ppf(0.995, R, 25.0 / 500)

bar_x = [-10, 510, 500, 510, -10, 0, -10]
bar_y = [sbc_high, sbc_high, sbc_mid, sbc_low, sbc_low, sbc_mid, sbc_high]

plot.fill(bar_x, bar_y, color="#DDDDDD", ec="#DDDDDD")
plot.plot([0, 500], [sbc_mid, sbc_mid], color="#999999", linewidth=2)

sbc_ranks = [x[1] for x in ensemble_output]

plot.hist(sbc_ranks, bins=[25 * x - 0.5 for x in range(21)],
          color=dark, ec=dark_highlight, zorder=3)

plot.gca().set_xlabel("Prior Rank")
plot.gca().set_xlim(-10, 510)
plot.gca().axes.get_yaxis().set_visible(False)

plot.show()

# Plot posterior sensitivities
z_scores = [x[2] for x in ensemble_output]
shrinkages = [x[3] for x in ensemble_output]

plot.scatter(shrinkages, z_scores, color=dark, alpha=0.2)
plot.gca().set_xlabel("Posterior Shrinkage")
plot.gca().set_xlim(0, 1)
plot.gca().set_ylabel("Posterior z-Score")
plot.gca().set_ylim(-5, 5)

plot.show()

############################################################
# POSTERIOR TO OBSERVATION
############################################################

############################################################
# 6. Fit the observations and evaluate
############################################################

data = pystan.read_rdump('workflow.data.R')

model = stan_utility.compile_model('fit_data_ppc.stan')
fit = model.sampling(data=data, seed=4838282)

# Check diagnostics
stan_utility.check_all_diagnostics(fit)

# Plot marginal posterior
params = fit.extract()

plot.hist(params['lambda'], bins = 25, color = dark, ec = dark_highlight)
plot.gca().set_xlabel("lambda")
plot.gca().axes.get_yaxis().set_visible(False)
plot.show()

############################################################
# 7. Analyze the Posterior Predictive Distribution
############################################################

max_y = 40
B = max_y + 1

bins = [b - 0.5 for b in range(B + 1)]

idxs = [ idx for idx in range(B) for r in range(2) ]
xs = [ idx + delta for idx in range(B) for delta in [-0.5, 0.5]]
      
obs_counts = numpy.histogram(data['y'], bins=bins)[0]
pad_obs_counts = [ obs_counts[idx] for idx in idxs ]

counts = [numpy.histogram(params['y_ppc'][n], bins=bins)[0] for n in range(4000)]
probs = [10, 20, 30, 40, 50, 60, 70, 80, 90]
creds = [numpy.percentile([count[b] for count in counts], probs)
         for b in range(B)]
pad_creds = [ creds[idx] for idx in idxs ]

plot.fill_between(xs, [c[0] for c in pad_creds], [c[8] for c in pad_creds],
                  facecolor=light, color=light)
plot.fill_between(xs, [c[1] for c in pad_creds], [c[7] for c in pad_creds],
                  facecolor=light_highlight, color=light_highlight)
plot.fill_between(xs, [c[2] for c in pad_creds], [c[6] for c in pad_creds],
                  facecolor=mid, color=mid)
plot.fill_between(xs, [c[3] for c in pad_creds], [c[5] for c in pad_creds],
                  facecolor=mid_highlight, color=mid_highlight)
plot.plot(xs, [c[4] for c in pad_creds], color=dark)

plot.plot(xs, pad_obs_counts, linewidth=2.5, color="white")
plot.plot(xs, pad_obs_counts, linewidth=2.0, color="black")

plot.gca().set_xlim([min(bins), max(bins)])
plot.gca().set_xlabel("y")
plot.gca().set_ylim([0, max(max(obs_counts), max([c[8] for c in creds]))])
plot.gca().set_ylabel("Posterior Predictive Distribution")

plot.show()

# The posterior predictive check indicates a serious
# excess of zeros above what we'd expect from a Poisson
# model.  Hence we want to expand our model to include
# zero-inflation and repeat the workflow.