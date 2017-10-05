/* An interface for fastText */
#include <streambuf>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <iomanip>

#include "interface.h"
#include "cpp/src/real.h"
#include "cpp/src/args.h"
#include "cpp/src/dictionary.h"
#include "cpp/src/matrix.h"
#include "cpp/src/vector.h"
#include "cpp/src/model.h"
#include "cpp/src/fasttext.h"

FastTextModel::FastTextModel(){}

bool quant_ = false;

void FastTextModel::setArgs(std::shared_ptr<Args> args)
{
    dim = args->dim;
    ws = args->ws;
    epoch = args->epoch;
    minCount = args->minCount;
    neg = args->neg;
    wordNgrams = args->wordNgrams;
    bucket = args->bucket;
    minn = args->minn;
    maxn = args->maxn;
    lrUpdateRate = args->lrUpdateRate;
    t = args->t;
    lr = args->lr;
}

void FastTextModel::setDictionary(std::shared_ptr<Dictionary> dict)
{
    _dict = dict;
}

void FastTextModel::setMatrix(std::shared_ptr<Matrix> input,
        std::shared_ptr<Matrix> output)
{
    _input_matrix = input;
    _output_matrix = output;
}

void FastTextModel::setModel(std::shared_ptr<Model> model)
{
    _model = model;
}

std::vector<std::vector<std::string>>
    FastTextModel::classifierPredictProb(std::string text, int32_t k)
{
    /* Hardcoded here; since we need this variable but the variable
     * is private in dictionary.h */
    const int32_t max_line_size = 1024;

    /* List of word ids */
    std::vector<int32_t> text_word_ids;
    std::vector<int32_t> word_hashes;
    std::istringstream iss(text);
    std::string token;

    /* We implement the same logic as Dictionary::getLine */
    std::uniform_real_distribution<> uniform(0, 1);
    while(_dict->readWord(iss, token)) {
        uint32_t h = _dict->hash(token);
        int32_t word_id = _dict->getId(token, h);

        if(word_id < 0) {
            entry_type type = _dict->getType(token);
            if (type == entry_type::word) word_hashes.push_back(h);
            continue;
        }

        entry_type type = _dict->getType(word_id);
        if (type == entry_type::word && !_dict->discard(word_id, uniform(_model->rng))) {
            text_word_ids.push_back(word_id);
            word_hashes.push_back(_dict->hash(token));
        }

        if(text_word_ids.size() > max_line_size) break;
    }
    _dict->addWordNgrams(text_word_ids, word_hashes, wordNgrams);

    std::vector<std::vector<std::string>> results;
    if(text_word_ids.size() > 0) {
        std::vector<std::pair<real, int32_t>> predictions;

        _model->predict(text_word_ids, k, predictions);
        for(auto it = predictions.cbegin(); it != predictions.cend(); it++) {
            std::vector<std::string> result;
            result.push_back(_dict->getLabel(it->second));

            /* We use string stream here instead of to_string, to make sure
             * that the string is consistent with std::cout from fasttext(1) */
            std::ostringstream probability_stream;
            probability_stream << exp(it->first);
            result.push_back(probability_stream.str());

            results.push_back(result);
        }
    }

    return results;
}

template <class cT, class traits = std::char_traits<cT> >
class basic_nullbuf: public std::basic_streambuf<cT, traits> {
    typename traits::int_type overflow(typename traits::int_type c)
    {
        return traits::not_eof(c); // indicate success
    }
};

void trainWrapper(int argc, char **argv, int silent)
{
    /* if silent > 0, the log from train() function will be supressed */

    std::vector<std::string> args(argv, argv + argc);

    if(silent > 0) {
        /* output file stream to redirect output from fastText library */
        std::streambuf* old_ofs = std::cout.rdbuf();
        std::streambuf* null_ofs = new basic_nullbuf<char>();
        std::cout.rdbuf(null_ofs);
        std::shared_ptr<Args> a = std::make_shared<Args>();
        a->parseArgs(args);
        FastText fasttext;
        fasttext.train(a);
        std::cout.rdbuf(old_ofs);
        delete null_ofs;
    } else {
        std::shared_ptr<Args> a = std::make_shared<Args>();
        a->parseArgs(args);
        FastText fasttext;
        fasttext.train(a);
    }
}


/* The logic is the same as FastText::loadModel, we roll our own
 * to be able to access data from args, dictionary etc since this
 * data is private in FastText class */
void loadModelWrapper(std::string filename, FastTextModel& model)
{
    std::ifstream ifs(filename, std::ifstream::binary);

    int32_t magic;
    int32_t version;
    ifs.read((char*)&(magic), sizeof(int32_t));
    ifs.read((char*)&(version), sizeof(int32_t));

    if (!ifs.is_open()) {
        std::cerr << "interface.cc: cannot load model file ";
        std::cerr << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    std::shared_ptr<Args> args = std::make_shared<Args>();
    std::shared_ptr<Dictionary> dict = std::make_shared<Dictionary>(args);
    std::shared_ptr<Matrix> input = std::make_shared<Matrix>();
    std::shared_ptr<Matrix> output = std::make_shared<Matrix>();
    std::shared_ptr<QMatrix> qinput = std::make_shared<QMatrix>();
    std::shared_ptr<QMatrix> qoutput = std::make_shared<QMatrix>();

    args->load(ifs);
    dict->load(ifs);

    bool quant_input;
    ifs.read((char*) &quant_input, sizeof(bool));
    if (quant_input) {
      quant_ = true;
      qinput->load(ifs);
    } else {
      input->load(ifs);
    }

    ifs.read((char*) &args->qout, sizeof(bool));
    if (quant_ && args->qout) {
        qoutput->load(ifs);
    } else {
        output->load(ifs);
    }

    std::shared_ptr<Model> model_p = std::make_shared<Model>(input, output, args, 0);
    model_p->quant_ = quant_;
    model_p->setQuantizePointer(qinput, qoutput, args->qout);

    if (args->model == model_name::sup) {
        model_p->setTargetCounts(dict->getCounts(entry_type::label));
    } else {
        model_p->setTargetCounts(dict->getCounts(entry_type::word));
    }
    ifs.close();

    /* save all data to FastTextModel */
    model.setArgs(args);
    model.setDictionary(dict);
    model.setMatrix(input, output);
    model.setModel(model_p);
}
