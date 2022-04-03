struct In {
    [[vk::location(0)]] float height: TEXCOORD0;
};

struct Out {
    [[vk::location(0)]] float4 height: TEXCOORD0;
};


Out main(In input) {
    Out output;

    output.height = float4(input.height);

    return output;
}