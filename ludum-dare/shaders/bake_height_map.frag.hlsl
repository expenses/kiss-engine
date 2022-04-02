struct In {
    [[vk::location(0)]] float height: TEXCOORD0;
};

struct Out {
    [[vk::location(0)]] float height: TEXCOORD0;
};


Out main(In input) {
    Out output;

    output.height = input.height;

    return output;
}