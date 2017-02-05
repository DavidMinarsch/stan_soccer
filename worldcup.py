"""
import os
os.chdir('/Users/davidminarsch/Desktop/PythonMLM/Worldcup_Example')
exec(open("worldcup.py").read())
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pystan
import seaborn as sns
sns.set(color_codes=True)
import scipy as sp
import pickle


teams = pd.read_csv("/Users/davidminarsch/Desktop/PythonMLM/Worldcup_Example/worldcup/soccerpowerindex.txt", sep=" ", header = None)
teams.columns = ['teams']
nteams = len(teams)
prior_score = np.arange(1,nteams+1)
prior_score = prior_score[::-1]
prior_score = (prior_score - np.mean(prior_score))/(2*np.std(prior_score))

data2012 = pd.read_csv("/Users/davidminarsch/Desktop/PythonMLM/Worldcup_Example/worldcup/worldcup2012.txt", sep=" ", header = None)
ngames = data2012.shape[0]

team1 = [teams[teams['teams']==x].index.values[0] for x in data2012[0]]
score1 = data2012[data2012.columns[1]]
team2 = [teams[teams['teams']==x].index.values[0] for x in data2012[2]]
score2 = data2012[data2012.columns[3]]

df = 7

# The model:
model_code = """
data {
  int<lower=0> nteams;
  int<lower=0> ngames;
  vector[nteams] prior_score;
  int<lower=0> team1[ngames];
  int<lower=0> team2[ngames];
  vector[ngames] score1;
  vector[ngames] score2;
  real<lower=0> df;
}
transformed data {
  vector[ngames] dif;
  dif = score1 - score2;
}
parameters {
  real b;
  real<lower=0> sigma_a;
  real<lower=0> sigma_y;
  vector[nteams] eta_a;
}
transformed parameters {
  vector[nteams] a;
  a = b*prior_score + sigma_a*eta_a;
}  
model {
  eta_a ~ normal(0,1);
  for (i in 1:ngames)
    dif[i] ~ student_t(df, a[team1[i]]-a[team2[i]], sigma_y);
}
"""

# The data dictionary:
data_dict = {'nteams': nteams, 'ngames': ngames, 'team1': team1,
  'score1': score1, 'team2': team2, 'score2': score2, 'prior_score': prior_score,
  'df': df}

# Fitting the model:
n_iter = 1000
chains = 4
fit = pystan.stan(model_code=model_code, data=data_dict, iter=n_iter, chains=chains)

print(fit)

#Alternative:
from pystan import StanModel
sm = StanModel(model_code=model_code)

# save it to the file 'worldcup.pkl' for later use:
with open('worldcup.pkl', 'wb') as f:
    pickle.dump(sm, f)

# load it again:
sm = pickle.load(open('worldcup.pkl', 'rb'))
fit = sm.sampling(data=data_dict, iter=n_iter, chains=chains)

print(fit)

# Plot results:
fit.plot()
plt.savefig('SummaryPlot.png')
plt.show()

# Boxplot:
params_a = pd.DataFrame(fit.extract(['a'],permuted=True)['a'])
params_a.rename(columns=lambda x: 'a'+ str(x), inplace=True)
params_a_mean = params_a.mean(axis=0)
params_a_se = params_a.std(axis=0)

sns.boxplot(params_a, orient="h")
plt.title('Team quality estimate')
plt.savefig('Boxplot.png')
plt.show()

# Plotting estimates and residuals:
#sns.barplot(data=params_a, orient="h", ci=95)
#plt.show()
x = np.arange(0,nteams)
y = np.array(params_a_mean)
plt.scatter(x,y)
yerr = np.array(params_a_se)
plt.errorbar(x,y,yerr=yerr, linestyle="None")
plt.title('Team quality estimate (+/- 1 s.e.)')
plt.xlabel('Team id')
plt.savefig('Errorbar.png')
plt.show()

#below is wrong...
expected_diff = params_a_mean.ix[team1].reset_index(drop=True) - params_a_mean.ix[team2].reset_index(drop=True)
params_sigma_y = pd.DataFrame(fit.extract(['sigma_y'],permuted=True)['sigma_y'])
interval_975 = np.median(scipy.stats.t.ppf(.975, df, loc=0, scale=1)*params_sigma_y)
def SignedSquare(a):
    return np.sign(a)*a**2
score_diff = score1 - score2
lower_error = score_diff - SignedSquare(expected_diff - interval_975)
upper_error = score_diff + SignedSquare(expected_diff + interval_975)

asymmetric_error = pd.concat([upper_error, lower_error], axis=1).as_matrix().T

x = np.arange(0,ngames)
y = score_diff
plt.scatter(x,y)
plt.errorbar(x,y,yerr=asymmetric_error, linestyle="None")
plt.title('Game score differentials\ncompared to 95% predictive interval from model\n')
plt.xlabel('Match id')
plt.savefig('GameScore.png')
plt.show()