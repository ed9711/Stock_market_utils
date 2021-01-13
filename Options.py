from bs4 import BeautifulSoup
import requests
import datetime
import time


def main():


    # data_url = "https://finance.yahoo.com/quote/SPY/options"
    dstamp = get_datestamp()
    data_url = "https://finance.yahoo.com/quote/SPY/options?date=" + dstamp
    print(data_url)
    data_html = requests.get(data_url).content
    # print(data_html)
    content = BeautifulSoup(data_html, "html.parser")
    # print(content)
    options_tables = []
    tables = content.find_all("table")
    for i in range(0, len(content.find_all("table"))):
        options_tables.append(tables[i])

    expiration = datetime.datetime.today().strftime("%Y - %m - %d")
    # calls
    calls = options_tables[0].find_all("tr")[1:]  # first row is header
    itm_calls = []
    otm_calls = []
    for call_option in calls:
        if "in-the-money Bgc($hoverBgColor)" in str(call_option):
            itm_calls.append(call_option)
        else:
            otm_calls.append(call_option)

    if itm_calls:
        # print(str(itm_calls[-1]))
        itm_call = itm_calls[-1]
    if otm_calls:
        otm_call = otm_calls[0]
    # print(str(itm_call) + " \n\n " + str(otm_call))

    itm_call_data = []
    for td in BeautifulSoup(str(itm_call), "html.parser").find_all("td"):
        itm_call_data.append(td.text)
    itm_call_info = {'contract': itm_call_data[0], 'strike': itm_call_data[2], 'last': itm_call_data[3],  'bid':
        itm_call_data[4], 'ask': itm_call_data[5], 'volume': itm_call_data[8], 'iv': itm_call_data[10]}
    print(itm_call_info)

    otm_call_data = []
    for td in BeautifulSoup(str(otm_call), "html.parser").find_all("td"):
        otm_call_data.append(td.text)
    otm_call_info = {'contract': otm_call_data[0], 'strike': otm_call_data[2], 'last': otm_call_data[3], 'bid':
        otm_call_data[4], 'ask': otm_call_data[5], 'volume': otm_call_data[8], 'iv': otm_call_data[10]}
    print(otm_call_info)

    # puts
    puts = options_tables[1].find_all("tr")[1:]
    itm_puts = []
    otm_puts = []
    for put_option in puts:
        if "in-the-money Bgc($hoverBgColor)" in str(put_option):
            itm_puts.append(put_option)
        else:
            otm_puts.append(put_option)
    itm_put = itm_puts[0]
    otm_put = otm_puts[-1]
    # print(str(itm_put) + " \n\n " + str(otm_put) + "\n\n")

    itm_put_data = []
    for td in BeautifulSoup(str(itm_put), "html.parser").find_all("td"):
        itm_put_data.append(td.text)
    itm_put_info = {'contract': itm_put_data[0], 'strike': itm_put_data[2], 'last': itm_put_data[3], 'bid':
        itm_put_data[4], 'ask': itm_put_data[5], 'volume': itm_put_data[8], 'iv': itm_put_data[10]}
    print(itm_put_info)

    otm_put_data = []
    for td in BeautifulSoup(str(otm_put), "html.parser").find_all("td"):
        otm_put_data.append(td.text)
    otm_put_info = {'contract': otm_put_data[0], 'strike': otm_put_data[2], 'last': otm_put_data[3], 'bid':
        otm_put_data[4], 'ask': otm_put_data[5], 'volume': otm_put_data[8], 'iv': otm_put_data[10]}
    print(otm_put_info)


def get_datestamp():
    options_url = "https://finance.yahoo.com/quote/SPY/options?date="
    today = int(time.time())
    # print(today)
    date = datetime.datetime.fromtimestamp(today)
    yy = date.year
    mm = date.month
    dd = date.day
    dd += 1
    options_day = datetime.datetime(yy, mm, dd, 16, 0, 0, 0)
    # print(options_day.timetuple())
    datestamp = int(time.mktime(options_day.timetuple()))
    # print(datestamp)
    # print(datetime.datetime.fromtimestamp(options_stamp))

    for i in range(0, 7):
        test_req = requests.get(options_url + str(datestamp)).content
        # print(options_url + str(datestamp))
        content = BeautifulSoup(test_req, "html.parser")
        # print(content)
        tables = content.find_all("table")
        if tables != []:
            print(datestamp)
            return str(datestamp)
        else:
            # print(“Bad datestamp!”)
            dd += 1
            options_day = datetime.datetime(yy, mm, dd, 16, 0, 0, 0)
            datestamp = int(time.mktime(options_day.timetuple()))
    return str(-1)


if __name__ == '__main__':
        main()
