import requests
from bs4 import BeautifulSoup as Soup
from pandas import DataFrame

BASE_URL = 'https://fantasyfootballcalculator.com/'


# EXERCISES
# 5.1.1
def scrape_ffc(scoring: str, nteams: int, year: int) -> DataFrame:
    if scoring.lower() not in ['ppr', 'half-ppr', 'standard']:
        raise ValueError('scoring must be either standard, ppr, or half')
    if year < 2018 and scoring.lower() == 'half-ppr':
        raise ValueError('half-ppr data unavailable prior to 2018')
    ffc_response = requests.get(BASE_URL + f'adp/{scoring.lower()}/{nteams}-team/all/{year}')

    adp_soup = Soup(ffc_response.text)

    # adp_soup is a nested tag, so call find_all on it
    tables = adp_soup.find_all('table')
    # get the adp table out of it
    adp_table = tables[0]
    rows = adp_table.find_all('tr')

    # put it in a function
    def parse_row(row):
        """
        Take in a tr tag and get the data out of it in the form of a list of
        strings.
        """
        return [str(x.string) for x in row.find_all('td')]

    # call function
    list_of_parsed_rows = [parse_row(row) for row in rows[1:]]

    # put it in a dataframe
    df = DataFrame(list_of_parsed_rows)
    df.head()

    # clean up formatting
    df.columns = ['ovr', 'pick', 'name', 'pos', 'team', 'adp', 'std_dev', 'high',
                  'low', 'drafted', 'graph']

    float_cols = ['adp', 'std_dev']
    int_cols = ['ovr', 'drafted']

    df[float_cols] = df[float_cols].astype(float)
    df[int_cols] = df[int_cols].astype(int)

    df.drop('graph', axis=1, inplace=True)
    return df

# 5.1.2
def scrape_ffc_with_link(scoring: str, nteams: int, year: int) -> DataFrame:
    if scoring.lower() not in ['ppr', 'half-ppr', 'standard']:
        raise ValueError('scoring must be either standard, ppr, or half')
    if year < 2018 and scoring.lower() == 'half-ppr':
        raise ValueError('half-ppr data unavailable prior to 2018')
    ffc_response = requests.get(BASE_URL + f'adp/{scoring.lower()}/{nteams}-team/all/{year}')

    adp_soup = Soup(ffc_response.text)

    # adp_soup is a nested tag, so call find_all on it
    tables = adp_soup.find_all('table')
    # get the adp table out of it
    adp_table = tables[0]
    rows = adp_table.find_all('tr')

    # put it in a function
    def parse_row_with_link(row):
        """
        Take in a tr tag and get the data out of it in the form of a list of
        strings.
        """
        row_out = []
        for x in row.find_all('td'):
            row_out.append(x.string)
        for link in row.find_all('a'):
            row_out.append('https://fantasyfootballcalculator.com' + str(link.get('href')))
        return row_out

    # call function
    list_of_parsed_rows = [parse_row_with_link(row) for row in rows[1:]]

    # put it in a dataframe
    df = DataFrame(list_of_parsed_rows)
    df.head()

    # clean up formatting
    df.columns = ['ovr', 'pick', 'name', 'pos', 'team', 'adp', 'std_dev', 'high',
                  'low', 'drafted', 'graph', 'player_link']

    float_cols = ['adp', 'std_dev']
    int_cols = ['ovr', 'drafted']

    df[float_cols] = df[float_cols].astype(float)
    df[int_cols] = df[int_cols].astype(int)

    df.drop('graph', axis=1, inplace=True)
    return df


# 5.1.3
def ffc_player_info(player_url: str) -> dict:
    ffc_response = requests.get(player_url)
    adp_soup = Soup(ffc_response.text)
    info_table = adp_soup.find_all('table')[0]
    draft_table = adp_soup.find_all('table')[1]

    info_fields = ['age', 'birthday', 'height', 'weight']
    info_d = dict(zip(info_fields, [str(x.string) for x in info_table.find_all('td')]))
    draft_fields = ['school', 'year', 'blank', 'details', 'team']
    draft_d = dict(zip(draft_fields, [str(x.string).strip() for x in draft_table.find_all('td')]))

    return {**info_d, **draft_d}
