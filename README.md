# llama-cpp-rs
An reimplementation of the parts of microsoft's [guidance](https://github.com/guidance-ai/guidance) that don't slow things down. Based on [llama.cpp](https://github.com/ggerganov/llama.cpp) with bindings in rust.

## Features

✅ Guarenteed LLM output formatting (see [formatting](#formatting))

✅ Dynamic prompt templates

✅ Model Quantization

✅ Fast (see [performace](#performace))

## Prompt storage.

You can store context on the filesystem if it will be reused, or keep the GRPC connection open to keep it in memory.

## Formatting

For a very simple example, assume you pass an LLM a transcript - you just sent the user a verification code, but you don't know if they've recived it yet, or if they are even able to access the 2fa device. You ask the user for the code - they respond and you prompt the LLM.

````
<transcript>
What is the users verification code?
```yaml
verification code: '
````

A tranditional solution (and the only solution offered by openai) is to give a stop condition of `'` you hope the llm to fills in a string and stops when it is done. You get *no control* on how it will respond. Without spending extra compute on a longer prompt you cannot specify that the code is 6 digits or what to output if it does not exist. And even with the longer prompt there is no guarentee it will be followed.

We do things differently by adding the ability to force an LLMs output to follow a regex and allowing bidirectional streaming.

- Given the regex `(true)|(false)` you can force a LLM to only respond with true or false.
- Given `([0-9]+)|(null)` you can extract a verification code that a user has given.

Combining the two leads to something like

````{ prompt: "<rest>verification code: '" }````

````{ generate: "(([0-9]+)|(null))'" }````

Which will always output the users verification code or `null`.

When combined with bidrirectional streaming we can do neat things, for example if the LLM yeilds a null `verification code`. We can send a second message asking for a `reason` (with the regex `(not arrived)|(unknown)|(device inaccessable)`).

### Comparisons

Guidance uses complex tempating sytnax. Dynamism is achvived though function calling and conditional statments in a handlebars like DSL. The function calling is a security nightmare (especially in a language as dynamic as python) and condional templating does not scale.

[lmql](https://lmql.ai/) uses a similar approach in that control flow stays in the "host" language, but it is a superset of python supported via decorators. Preformance is difficult to control and near impossible to use in a concurrent setting such as a web server.

We instead stick the LLM on a GPU (or many if resources are required) and call to it using GRPC.

Dynamism is achived in the client code (where it belongs) by streaming messages back and forth between the client and `llama-cpp-rpc` with minimal overhead.

## Performace

Numbers are run on a 3090 running a finetuned 7b Minseral model (unquantized). With quantization we can run state of the art 70b models on consumer hardware.

||Remote Hosting|FS context storage|concurrency|raw tps|guided tps|
|----|----|----|----|----|----|
|Llama-cpp-rpc|✅|✅|✅|65|56||
|Guidance|❌|❌|❌|30|5||
|LMQL|❌|❌|❌|30|10||

## Dependencies

### Ubuntu

```bash
sudo apt install -y curl libssl-dev libclang-dev pkg-config cmake git protobuf-compiler
```
