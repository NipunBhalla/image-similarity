# Image Similarity

Flask API which takes a text input and returns the most relevant documents.

## Installation
To run the code locally, clone the repo and run the following commands.

```bash
cd image-similarity
python application.py --port 5000
```

You can also use the docker image from Docker Hub directly: https://hub.docker.com/r/nipunbhalla/problem1

```bash
docker pull nipunbhalla/problem1
```


## Usage
The API accepts POST requests and requires 3 form fileds, OR 1 form field and 2 files:

| param |  Description |
| ------ | ------ |
| type | 0 for image URL mode. 1 for image upload mode |
| img1 | If type=0, send img1 as form field with IMAGE URL. If type=1, send img1 as file field. |
| img2 | If type=0, send img2 as form field with IMAGE URL. If type=1, send img2 as file field. |


## Response
API returns a JSON object with score. Similarity score ranges from 0-100, where 0 means no match and 100 means exact match.

```json
[
  {
    "score": 69.420
  }
]
```


## License
[MIT](https://choosealicense.com/licenses/mit/)