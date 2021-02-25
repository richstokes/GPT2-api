HTTP API that lets you make (concurrent) GPT-2 requests and receive the generated text output as JSON.

## Build n Run
`./bnr.sh`  

Uses a docker image to remove the complexity of getting a working python+tensorfloww environment working locally. 


## You can then send a request with 
```
curl --request POST --data '{"wp":"Never gonna give you up"}' http://localhost:2666/wp
```