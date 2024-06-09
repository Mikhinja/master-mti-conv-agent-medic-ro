# master-mti-conv-agent-medic-ro

Data scraped from [https://www.sfatulmedicului.ro/comunitate](https://www.sfatulmedicului.ro/comunitate).

## Steps and scripts

### Data gathering
The `take_medical_conversations.py` script crawls the site and downloads each post in a `json` file under data/raw/&lt;category&gt;.

The `sanitize_medical_conversations.py` script sanitizes further, breaks down words, takes keywords by eliminating stopwords, identifies censored workds and tries to salvage some of them that have been empirically proven to be safe to replace, and adds some processed information such as the number of censored words
