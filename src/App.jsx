import React, { useState, useEffect } from 'react';
import { pipeline } from '@xenova/transformers';

// Dot-product cosine similarity
function cosineSimilarity(a, b) {
  const dot = a.reduce((sum, ai, i) => sum + ai * b[i], 0);
  const magA = Math.sqrt(a.reduce((sum, ai) => sum + ai * ai, 0));
  const magB = Math.sqrt(b.reduce((sum, bi) => sum + bi * bi, 0));
  return dot / (magA * magB);
}

// Average token embeddings into one vector
function averageTokens(tokens2d) {
  const dim = tokens2d[0].length;
  const sum = new Array(dim).fill(0);
  tokens2d.forEach(tok => tok.forEach((v, i) => (sum[i] += v)));
  return sum.map(v => v / tokens2d.length);
}

// Extract flat JS array and token count from output
function parseOutput(out) {
  let vec;
  let tokenCount = 0;
  if (Array.isArray(out)) {
    const batchItem = out[0];
    if (Array.isArray(batchItem[0])) {
      vec = averageTokens(batchItem);
      tokenCount = batchItem.length;
    } else {
      vec = batchItem;
      tokenCount = 1;
    }
  } else if (out.data && Array.isArray(out.dims)) {
    const { data, dims } = out;
    if (dims.length === 3) {
      const [, seqLen, dim] = dims;
      tokenCount = seqLen;
      const tokens = [];
      for (let i = 0; i < seqLen; i++) {
        const start = i * dim;
        tokens.push(Array.from(data.slice(start, start + dim)));
      }
      vec = averageTokens(tokens);
    } else if (dims.length === 2) {
      const [batch, dim] = dims;
      tokenCount = batch;
      const sum = new Array(dim).fill(0);
      for (let b = 0; b < batch; b++) {
        for (let j = 0; j < dim; j++) sum[j] += data[b * dim + j];
      }
      vec = sum.map(v => v / batch);
    }
  } else {
    throw new Error('Unsupported pipeline output format');
  }
  return { vec, tokenCount };
}

export default function App() {
  const [model, setModel] = useState(null);
  const [loadingModel, setLoadingModel] = useState(true);
  const [error, setError] = useState(null);

  const [userText, setUserText] = useState('');
  const [expectedText, setExpectedText] = useState('');
  const [userTokens, setUserTokens] = useState(0);
  const [expectedTokens, setExpectedTokens] = useState(0);
  const [loading, setLoading] = useState(false);
  const [similarity, setSimilarity] = useState(null);
  const [vec1, setVec1] = useState(null);
  const [vec2, setVec2] = useState(null);

  useEffect(() => {
    pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2')
      .then(pipe => setModel(() => pipe))
      .catch(err => setError('Model load failed: ' + err.message))
      .finally(() => setLoadingModel(false));
  }, []);



  const compareTexts = async () => {
    if (!model) return;
    setLoading(true);
    setError(null);
    setSimilarity(null);
    setVec1(null);
    setVec2(null);

    try {
      const [out1, out2] = await Promise.all([
        model(userText), model(expectedText)
      ]);
      const { vec: v1, tokenCount: t1 } = parseOutput(out1);
      const { vec: v2, tokenCount: t2 } = parseOutput(out2);
      setVec1(v1);
      setVec2(v2);
      setUserTokens(t1);
      setExpectedTokens(t2);
      setSimilarity(cosineSimilarity(v1, v2));
    } catch (err) {
      setError('Error generating embeddings: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6 space-y-6">
      {/* Header */}
      <div className="p-4 bg-gray-800 rounded">
        <h1 className="text-3xl font-bold">Embedding Playground</h1>
      </div>

      {/* User Input Section */}
      <div className="grid grid-cols-4 gap-4">
        <div className="col-span-3 p-4 bg-gray-800 rounded space-y-2">
          <label className="block font-semibold">User Input</label>
          <textarea
            value={userText}
            onChange={e => setUserText(e.target.value)}
            className="w-full bg-gray-700 text-white border border-gray-600 p-3 rounded"
            rows={3}
          />
          <div className="text-sm text-gray-400">Tokens: {userTokens}</div>
          {userText.trim() && (
            <>
              <h3 className="font-semibold">Python code</h3>
              <pre className="bg-gray-700 p-2 rounded text-xs font-mono text-blue-200 whitespace-pre-wrap mb-2">
{`from transformers import pipeline
embed = pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2")
vec1 = embed(user_text)`}
              </pre>
            </>
          )}
        </div>
        <div className="col-span-1 p-4 bg-gray-800 rounded">
          <h3 className="font-semibold mb-2">Input Vector</h3>
          {vec1 && (
            <div className="max-h-40 overflow-y-auto bg-gray-700 p-2 rounded text-xs font-mono text-green-200 whitespace-pre-wrap">
              {JSON.stringify(vec1, null, 2)}
            </div>
          )}
        </div>
      </div>

      {/* Expected Text Section */}
      <div className="grid grid-cols-4 gap-4">
        <div className="col-span-3 p-4 bg-gray-800 rounded space-y-2">
          <label className="block font-semibold">Expected Text</label>
          <textarea
            value={expectedText}
            onChange={e => setExpectedText(e.target.value)}
            className="w-full bg-gray-700 text-white border border-gray-600 p-3 rounded"
            rows={3}
          />
          <div className="text-sm text-gray-400">Tokens: {expectedTokens}</div>
          {expectedText.trim() && (
            <>
              <h3 className="font-semibold">Python code</h3>
              <pre className="bg-gray-700 p-2 rounded text-xs font-mono text-blue-200 whitespace-pre-wrap mb-2">
{`from transformers import pipeline
embed = pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2")
vec2 = embed(expected_text)

from scipy.spatial.distance import cosine
sim = 1 - cosine(vec1, vec2)`}
              </pre>
            </>
          )}
        </div>
        <div className="col-span-1 p-4 bg-gray-800 rounded">
          <h3 className="font-semibold mb-2">Expected Vector</h3>
          {vec2 && (
            <div className="max-h-40 overflow-y-auto bg-gray-700 p-2 rounded text-xs font-mono text-green-200 whitespace-pre-wrap">
              {JSON.stringify(vec2, null, 2)}
            </div>
          )}
        </div>
      </div>

      {/* Action */}
      <div className="p-4 bg-gray-800 rounded">
        <button
          onClick={compareTexts}
          disabled={loadingModel || loading || !userText.trim() || !expectedText.trim()}
          className="bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white px-5 py-2 rounded"
        >
          {loadingModel ? 'Loading model...' : loading ? 'Comparing...' : 'Compare Similarity'}
        </button>
      </div>

      {/* Result */}
      {similarity !== null && (
        <div className="p-4 bg-gray-800 rounded space-y-2">
          <div className="flex items-center">
            <span>Similarity:</span>
            <div className="relative group ml-2 inline-block">
              <span className="cursor-pointer font-bold">?</span>
              <div className="absolute left-0 bottom-full mb-1 invisible group-hover:visible bg-gray-700 text-xs text-white p-2 rounded whitespace-nowrap">
                Good: ≥0.85; OK: 0.5–0.85; Bad: &lt;0.5
              </div>
            </div>
          </div>
          {/* Re-added similarity bar */}
          <div className="h-4 bg-gray-700 rounded mb-2">
            <div
              className={`h-4 rounded ${similarity >= 0.85 ? 'bg-green-500' : similarity >= 0.5 ? 'bg-orange-400' : 'bg-red-500'}`}
              style={{ width: `${(similarity * 100).toFixed(1)}%` }}
            />
          </div>
          <div className="text-sm">{similarity.toFixed(4)}</div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="p-4 bg-red-700 rounded text-white">{error}</div>
      )}
    </div>
  );
}
