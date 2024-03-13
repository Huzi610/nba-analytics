import argparse
import json
import warnings
from datetime import date
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gamma, exponpow, lognorm, cauchy, genhyperbolic
from sklearn.mixture import BayesianGaussianMixture
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup

warnings.filterwarnings('ignore')


def z(data):
    mean = np.mean(data)
    std = np.std(data)

    return (data - mean) / std

def mix_norm_cdf(x, weights, means, covars):
    mcdf = 0.0
    for i in range(len(weights)):
        mcdf += weights[i] * stats.norm.cdf(x, loc=means[i], scale=covars[i])
    return mcdf

def calc_h_score_3(data, clf):
    probs = mix_norm_cdf(data, means=clf.means_.flatten(), covars=clf.covariances_.flatten(), weights=clf.weights_)
    h_scores = stats.norm.ppf(probs)
    return h_scores

def calc_h_score(data, cat):
    probs = cat_dist_dict[cat].cdf(data[cat].values)
    h_scores = stats.norm.ppf(probs)
    if cat == 'to/g':
        return -h_scores
    return h_scores

def calc_perc_h_score(data, cat):
    cat_a = cat.replace('%', 'a/g')
    shots_made = data[cat].values * data[cat_a].values
    perc_avg = shots_made.sum() / data[cat_a].sum()
    impact = (data[cat] - perc_avg) * data[cat_a]

    probs = cat_dist_dict[cat].cdf(impact)

    h_scores = stats.norm.ppf(probs)

    return h_scores


def extract_projections_from_page():
    cj = {'RotoMonsterUserId': 't6+HU/vXxy5wnktE6HvldJ08rjAUwU/Ot2lkzomRRzo=',
          'ASP.NET_SessionId': 'jp3ronkkgivdab1f0o5xpkca'
          }
    url = 'https://basketballmonster.com/playerrankings.aspx'
    # url = 'https://basketballmonster.com/projections.aspx'
    r = requests.get(url, cookies=cj)

    soup = BeautifulSoup(r.text, 'lxml')

    # Obtain information from tag <table>
    table1 = soup.find('table', {"class": 'table-bordered table-hover table-sm base-td-small datatable ml-0'})

    df = pd.read_html(str(table1))[0]

    columns = ['Rank',
               # 'Y!Adp',
               'Value', 'Name', 'p/g', '3/g',
               'r/g', 'a/g', 's/g', 'b/g', 'fg%', 'ft%', 'fta/g', 'fga/g', 'to/g']

    curr_df = df[columns]

    curr_df = curr_df[~curr_df.Rank.str.startswith('Ra')]

    # curr_df = curr_df.astype(float)

    curr_df = curr_df.apply(lambda i: i.apply(lambda x: float(x) if str(x).replace('.', '', 1).isdigit() else x))

    curr_df = curr_df.reset_index()

    return curr_df


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='CARUSO Table Maker')

    # parser.add_argument('--filename', '--f', type=str, required=True, help='Excel file with projections or rankings')
    parser.add_argument('--turnover', '--t', type=bool, help='Include turnover or not', default=False)

    args = parser.parse_args()
    # print(args)

    # filename = args.filename
    turnovers_flag = args.turnover

    dist_dict = {'gamma': gamma,
                 'lognorm': lognorm,
                 'exponpow': exponpow,
                 'cauchy': cauchy,
                 'genhyperbolic': genhyperbolic
                 }

    cat_dist_dict = {}

    player = {}

    cnt = 1

    plot = False


    with open('9cat_parameters.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith('#'):
                continue
            else:
                cat, str_dict, error = line.replace(' ', '').split('_')
                str_dict = str_dict.replace("'", '"')
                par_dict = json.loads(str_dict)
                dist = list(par_dict.keys())[0]

                cat_dist_dict[cat] = dist_dict[dist](*list(par_dict[dist].values()))

    clf = BayesianGaussianMixture(n_components=2)

    clf.means_ = np.array([[1.66894348, 0.07495573]])
    clf.covariances_ = np.array([[0.7040092, 0.0212879]])
    clf.weights_ = np.array([0.82865659, 0.17134341])

    # df_2023 = pd.read_excel(filename)
    df_2023 = extract_projections_from_page()

    h_score_df = df_2023.copy()

    cats = ['p/g',
            '3/g',
            'r/g',
            'a/g',
            's/g',
            'b/g',
            'to/g',
            'ft%',
            'fg%']

    if turnovers_flag:
        cats.append('to/g')

    for cat in cats:
        if '%' in cat:
            h_score_df[f'{cat}_hV'] = calc_perc_h_score(h_score_df, cat)
        elif '3' in cat:
            h_score_df['3/g_hV'] = calc_h_score_3(h_score_df['3/g'], clf)
        else:
            h_score_df[f'{cat}_hV'] = calc_h_score(h_score_df, cat)

    h_scores_columns = []

    for column_name in cats:
        column_name = f'{column_name}_hV'
        h_scores_columns.append(column_name)

    h_score_df['CARUSO'] = h_score_df[h_scores_columns].mean(axis=1)


    columns_to_keep = ['Rank',
                       'Value',
                       'CARUSO',
                       'Name',
                       # 'Y!Adp'
                       ]

    columns_to_keep.extend(h_scores_columns)
    columns_to_keep.extend(cats)
    columns_to_keep.extend(['fta/g', 'fga/g'])

    h_score_df = h_score_df[columns_to_keep]

    # h_score_df = h_score_df.drop(h_score_df[h_score_df['Y!Adp'] <= 0].index)
    # h_score_df = h_score_df.drop(h_score_df[h_score_df['Y!Adp'] >= 150].index)

    h_score_df = h_score_df.sort_values(by=['CARUSO'], ascending=False)
    h_score_df = h_score_df.reset_index(drop=True)

    h_score_df['CARUSO Rank'] = h_score_df.index + 1

    columns = list(h_score_df.columns)

    rk_chg = columns.pop()
    columns.insert(1, rk_chg)

    h_score_df['Rank Change'] = h_score_df['Rank'] - (h_score_df.index + 1)

    columns = list(h_score_df.columns)

    rk_chg = columns.pop()
    columns.insert(1, rk_chg)

    h_score_df = h_score_df[columns]

    rk_chg = columns.pop()
    columns.insert(2, rk_chg)

    h_score_df = h_score_df[columns]

    today = date.today()

    output_name = f'Excels/caruso_{today}.xlsx'
    h_score_df.to_excel(output_name)

    print(f'Created File {output_name}')

    players = []
    with open('keeper_players.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            players.append(line.rstrip())

    players_df = pd.DataFrame()

    for player in players:
        player_row = h_score_df[h_score_df['Name'] == player]
        # player_row = pd.DataFrame(player_row)
        # print(player_row)
        players_df = pd.concat([players_df, player_row])

    players_df.to_excel(f'Excels/huzi_keeperchina_{today}_caruso.xlsx')


