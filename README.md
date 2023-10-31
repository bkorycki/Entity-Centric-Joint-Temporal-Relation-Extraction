# An Entity-Centric Model for Joint Temporal Relation Extraction

## Input data format
Input data is a JSON file in which each item is a document. For example:
```
{
  "doc_id": "doc0001", # unique ID for each document
  "title": "",
  "body": "I was diagnosed with bipolar 2 years ago and started taking Lamictal. Last April, my psych switched me to Abilify because of issues with my blood tests. I didn't have any side effects with the lamictal but now I feel just super antsy and restless-- has anyone else had this issue?",
  "dct": "2019-06-22", 	# Document Creation Time
  "meds": { # list of medication entities and their respective mentions
	"196502": [
		{
		"mention_id": "mention_0_196502"
		"string": "Lamictal",
		"entity_id": "196502", # RxCUI code
		"type": "med",
		"source": 1, 		# 0 for title, 1 for body
		"span": [60, 68] # Mention's token span in text
		},
		{
		"mention_id": "mention_1_196502"
		"string": "lamictal",
		"entity_id": "196502", 
		"type": "med",
		"source": 1, 		
		"span": [193, 201]
		}
	],
	"352393": [
		{
		"mention_id": "mention_0_352393"
		"string": "Abilify",
		"entity_id": "352393", 
		"type": "med",
		"source": 1, 		
		"span": [106, 113] 
		}
	]
  },
  "dates": # list of date-time 'entities' grouped by equal value
	"20170622000000": [
		{
		"mention_id": "mention_0_20170622000000"
		"string": " 2 years ago",
		"value": "2017-06-22 00:00:00"
		"entity_id": "20170622000000",
		"type": "date",
		"source": 1, 		# 0 for title, 1 for body
		"span": [29, 40] # Mention's token span in text
		}
	],
	"20190401000000": [
		{
		"mention_id": "mention_0_20190401000000"
		"string": "Last April",
		"value": "2019-04-01 00:00:00"
		"entity_id": "20190401000000", 
		"type": "med",
		"source": 1, 		
		"span": [70, 80] 
		}
	]
  },
  "labels": { # temporal relations
	"196502": { 
		"20170622000000": "start",
		"20190401000000": "stop,
		"DCT": "after"
	} "352393": { 
		"20170622000000": "before",
		"20190401000000": "start,
		"DCT": "on"
	}	
  },
  "user_id": "user01", 
  "subreddit": "Bipolar", # Name of subreddit the post was scraped from
```

## Usage
```
python src/relation_extraction.py --config [config_name]
```

## Versions
- `python3.8.16`
- `torch.2.0.1`
- CUDA 11.7
