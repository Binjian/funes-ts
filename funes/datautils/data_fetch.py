#!/usr/bin/env python
# coding: utf-8

# This script is to download data from elastic platform by some features(car number,time range)
# data will be store in json file

# system packages
import json
from datetime import datetime
from collections import Counter
from datetime import datetime

# third-party packages
from typing import MutableSet
from elasticsearch import Elasticsearch


def main():
    # car vin number to search
    carNo1 = "HMZABAAHXMF014505"
    # carNo2 = "HMZABAAHXMF018277"
    # json file to store data
    filename = "all-info/car111" + carNo1 + ".json"
    # time range
    start_time = "2022.1.23 00:00:00"
    end_time = "2022.3.1 00:00:00"
    # TODO change time range
    # convert time range to timestamp
    start_time = int(
        datetime.strptime(start_time, "%Y.%m.%d %H:%M:%S").timestamp() * 1000
    )
    end_time = int(datetime.strptime(end_time, "%Y.%m.%d %H:%M:%S").timestamp() * 1000)
    # define soc range
    start_soc = 80
    end_soc = 100
    # define elastic address & port and authentication
    es = Elasticsearch(
        hosts=[
            "http://10.0.80.13:9200",
            "http://10.0.80.14:9200",
            "http://10.0.80.15:9200",
        ],
        http_auth=("elastic", "123456"),
    )
    # start fetch
    print("start fetch data")
    # define search range and filter
    body = {
        "query": {
            "bool": {
                # must match car vin number and charge state
                "must": [
                    {"match": {"vin": carNo1}},
                    {"match": {"canData.data.VCU_BMSChargeState": 3}},
                ],
                # filt the search time range
                "filter": [
                    {
                        "range": {
                            "travelTime": {
                                "gte": start_time,
                                "lte": end_time,
                            }
                        }
                    },
                    {
                        "range": {
                            "canData.data.BMSSOC": {
                                "gte": start_soc,
                                "lte": end_soc,
                            }
                        }
                    },
                ],
            }
        },
        # sort data via travel time and by order of asc
        "sort": [{"travelTime": {"order": "asc"}}],
    }
    # index is the project name, size is defined size of fetched data
    res = es.search(
        index="tsp-vehicle-log3*", body=body, size=500000, request_timeout=300000
    )
    # print(res)                                                                       #size is the number of data to get.
    print("end fetch data")

    # save result to json file
    with open(filename, "w") as file_obj:
        json.dump(res["hits"]["hits"], file_obj)  # save

    # finish saving
    print("success")


if __name__ == "__main__":
    main()
