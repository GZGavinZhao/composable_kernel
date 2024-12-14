#pragma once

TYPED_TEST(TestCkTileGroupedGemm, Basic)
{
    const int group_count = 16;
    std::vector<int> Ms;
    std::vector<int> Ns;
    std::vector<int> Ks;
    std::vector<int> stride_As;
    std::vector<int> stride_Bs;
    std::vector<int> stride_Cs;

    for(int i = 0; i < group_count; i++)
    {
        Ms.push_back(256 + 256 * i);
        Ns.push_back(128 + 128 * i);
        Ks.push_back(128 + 64 * i);

        stride_As.push_back(Ks[i]);
        stride_Bs.push_back(Ks[i]);
        stride_Cs.push_back(Ns[i]);
    }

    this->Run(Ms, Ns, Ks, stride_As, stride_Bs, stride_Cs, group_count);
}