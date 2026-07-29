/*
 * chirp-sync encoder -- the transmit half of the Python library, in plain JS.
 *
 * Every step here has to be bit-exact with chirpsync/{codec,fec,css,frame}.py,
 * because a chirp emitted from a phone gets decoded by the Python parser later.
 * tests/test_js_parity.py renders waveforms through this file under Node and
 * decodes them with Python to prove it.
 *
 * Runs unmodified in a browser and in Node (see the export shim at the bottom).
 * The plain JavaScript implementation is designed for long-lived browser use.
 */
(function (root) {
  'use strict';

  var FS_WORK = 16000;
  var TAKE_ID_BITS = 40;
  var TAKE_ID_BYTES = TAKE_ID_BITS / 8;
  var PAYLOAD_BYTES = TAKE_ID_BYTES + 2;
  var B32 = '0123456789ABCDEFGHJKMNPQRSTVWXYZ';

  var PROFILES = {
    fast:     { name: 'fast',     sf: 7, bw: 4000, fLow: 1000, preamble: 8, sfd: 2 },
    balanced: { name: 'balanced', sf: 8, bw: 4000, fLow: 1000, preamble: 8, sfd: 2 },
    robust:   { name: 'robust',   sf: 9, bw: 4000, fLow: 1000, preamble: 8, sfd: 2 }
  };

  // --- small helpers ---------------------------------------------------------

  function symbolTime(profile, sf) { return (1 << sf) / profile.bw; }

  function takeIdToStr(bytes) {
    // bytes: 5 bytes, big endian, 40 bits -> 8 Crockford base32 characters.
    var out = '';
    var acc = 0, bits = 0;
    for (var i = 0; i < 5; i++) {
      acc = (acc * 256) + bytes[i];
      bits += 8;
      while (bits >= 5) {
        bits -= 5;
        var shift = Math.pow(2, bits);
        var idx = Math.floor(acc / shift) % 32;
        out += B32[idx];
      }
    }
    return out;
  }

  function randomTakeId() {
    var bytes = new Uint8Array(5);
    if (root.crypto && root.crypto.getRandomValues) {
      root.crypto.getRandomValues(bytes);
    } else {
      for (var i = 0; i < 5; i++) bytes[i] = Math.floor(Math.random() * 256);
    }
    return bytes;
  }

  // --- checksums -------------------------------------------------------------

  function crc16(data) {
    var crc = 0xFFFF;
    for (var i = 0; i < data.length; i++) {
      crc ^= data[i] << 8;
      for (var b = 0; b < 8; b++) {
        crc = (crc & 0x8000) ? ((crc << 1) ^ 0x1021) & 0xFFFF : (crc << 1) & 0xFFFF;
      }
    }
    return crc & 0xFFFF;
  }

  // --- whitening / FEC / interleaving ---------------------------------------

  function pn9(n) {
    var state = 0x1FF;
    var out = new Uint8Array(n);
    for (var i = 0; i < n; i++) {
      var byte = 0;
      for (var b = 0; b < 8; b++) {
        byte |= (state & 1) << b;
        var fb = ((state & 1) ^ ((state >> 5) & 1)) & 1;
        state = (state >> 1) | (fb << 8);
      }
      out[i] = byte;
    }
    return out;
  }

  function whiten(data) {
    var mask = pn9(data.length);
    var out = new Uint8Array(data.length);
    for (var i = 0; i < data.length; i++) out[i] = data[i] ^ mask[i];
    return out;
  }

  var K = 7, POLYS = [0o171, 0o133, 0o165], TAIL = K - 1;

  function parity(x) {
    x ^= x >> 4; x ^= x >> 2; x ^= x >> 1;
    return x & 1;
  }

  function convEncode(bits) {
    var padded = new Uint8Array(bits.length + TAIL);
    padded.set(bits, 0);
    var out = new Uint8Array(codedLength(bits.length));
    var state = 0;
    var at = 0;
    for (var i = 0; i < padded.length; i++) {
      var reg = (padded[i] << (K - 1)) | state;
      for (var p = 0; p < POLYS.length; p++) {
        if (p < 2 || i % 3 !== 2) out[at++] = parity(reg & POLYS[p]);
      }
      state = reg >> 1;
    }
    return out;
  }

  function gcd(a, b) { while (b) { var t = a % b; a = b; b = t; } return a; }

  function interleaveIndices(n) {
    if (n < 3) return null;
    var stride = Math.max(1, Math.floor(n / 1.6180339887));
    while (gcd(stride, n) !== 1) {
      stride += 1;
      if (stride >= n) { stride = 1; break; }
    }
    var idx = new Int32Array(n);
    for (var i = 0; i < n; i++) idx[i] = (i * stride) % n;
    return idx;
  }

  function interleave(bits) {
    var idx = interleaveIndices(bits.length);
    if (!idx) return bits;
    var out = new Uint8Array(bits.length);
    for (var i = 0; i < bits.length; i++) out[i] = bits[idx[i]];
    return out;
  }

  // --- block -> symbols ------------------------------------------------------

  function bytesToBits(data) {
    var bits = new Uint8Array(data.length * 8);
    for (var i = 0; i < data.length; i++) {
      for (var b = 0; b < 8; b++) bits[i * 8 + b] = (data[i] >> (7 - b)) & 1;
    }
    return bits;
  }

  function encodeBlock(data, sf) {
    var coded = convEncode(bytesToBits(whiten(data)));
    var nsym = Math.ceil(coded.length / sf);
    var padded = new Uint8Array(nsym * sf);
    padded.set(coded, 0);
    var il = interleave(padded);
    var symbols = new Int32Array(nsym);
    for (var s = 0; s < nsym; s++) {
      var v = 0;
      for (var j = 0; j < sf; j++) v = (v << 1) | il[s * sf + j];
      symbols[s] = v ^ (v >>> 1);   // Gray encode
    }
    return symbols;
  }

  function blockSymbols(nBytes, sf) {
    return Math.ceil(codedLength(nBytes * 8) / sf);
  }

  function codedLength(nInfo) {
    var steps = nInfo + TAIL;
    return 2 * steps + 2 * Math.floor(steps / 3) + Math.min(steps % 3, 2);
  }

  // --- payload ---------------------------------------------------------------

  function buildPayload(opts) {
    var takeBytes = opts.takeId || randomTakeId();
    if (!(takeBytes instanceof Uint8Array)) takeBytes = Uint8Array.from(takeBytes);
    if (takeBytes.length !== TAKE_ID_BYTES) {
      throw new Error('takeId must contain exactly ' + TAKE_ID_BYTES + ' bytes');
    }
    var bytes = new Uint8Array(PAYLOAD_BYTES);
    bytes.set(takeBytes, 0);
    var c16 = crc16(takeBytes);
    bytes[TAKE_ID_BYTES] = (c16 >> 8) & 0xFF;
    bytes[TAKE_ID_BYTES + 1] = c16 & 0xFF;

    return {
      take: takeIdToStr(takeBytes),
      takeBytes: takeBytes,
      bytes: bytes
    };
  }

  // --- waveform --------------------------------------------------------------

  function frameItems(payload, profile) {
    var items = [];
    var i;
    for (i = 0; i < profile.preamble; i++) items.push(['up', 0, profile.sf]);
    for (i = 0; i < profile.sfd; i++) items.push(['down', 0, profile.sf]);
    var dataSyms = encodeBlock(payload.bytes, profile.sf);
    for (i = 0; i < dataSyms.length; i++) items.push(['up', dataSyms[i], profile.sf]);
    var total = profile.preamble + profile.sfd + dataSyms.length;
    return {
      items: items,
      layout: {
        dataSymbols: dataSyms.length,
        duration: total * symbolTime(profile, profile.sf)
      }
    };
  }

  /* Render to a Float32Array.
   *
   * Symbol boundaries are computed in seconds and each sample asks which symbol
   * it falls in, rather than each symbol being rendered as its own rounded
   * number of samples.  A symbol is 2^SF/BW seconds, which is a whole number of
   * samples at 48 kHz but not at 44.1 kHz, so rounding per symbol would let the
   * burst drift against the receiver's grid -- about a chip and a half by the
   * end of a frame, enough to lose its tail.
   *
   * Phase is then integrated trapezoidally, which is exact for the piecewise
   * linear ramps a chirp is made of and stays continuous at the sweep's wrap
   * point and across symbol boundaries. */
  function render(items, profile, sampleRate, fadeSeconds) {
    var k;
    var bounds = new Float64Array(items.length + 1);
    for (k = 0; k < items.length; k++) {
      bounds[k + 1] = bounds[k] + symbolTime(profile, items[k][2]);
    }
    var total = Math.round(bounds[items.length] * sampleRate);

    var freq = new Float64Array(total);
    var at = 0;
    for (var j = 0; j < total; j++) {
      var t = j / sampleRate;
      while (at < items.length - 1 && t >= bounds[at + 1]) at++;
      var kind = items[at][0], value = items[at][1], sfk = items[at][2];
      var span = bounds[at + 1] - bounds[at];
      var local = (t - bounds[at]) / span;
      var frac = (local + value / (1 << sfk)) % 1.0;
      if (kind === 'down') frac = (1.0 - frac) % 1.0;
      freq[j] = profile.fLow + profile.bw * frac;
    }

    var out = new Float32Array(total);
    var phase = 0.0;
    var twoPiOverFs = 2 * Math.PI / sampleRate;
    for (var j = 0; j < total; j++) {
      if (j > 0) phase += twoPiOverFs * 0.5 * (freq[j] + freq[j - 1]);
      out[j] = Math.cos(phase);
    }

    var nf = Math.round((fadeSeconds == null ? 0.004 : fadeSeconds) * sampleRate);
    if (nf > 0 && total > 2 * nf) {
      for (var f = 0; f < nf; f++) {
        var ramp = 0.5 - 0.5 * Math.cos(Math.PI * f / nf);
        out[f] *= ramp;
        out[total - 1 - f] *= ramp;
      }
    }
    return out;
  }

  function generate(opts) {
    opts = opts || {};
    var profile = PROFILES[opts.profile || 'fast'];
    if (!profile) throw new Error('unknown profile ' + opts.profile);
    var sampleRate = opts.sampleRate || 48000;
    var payload = buildPayload(opts);
    var built = frameItems(payload, profile);
    var burst = render(built.items, profile, sampleRate, opts.fade);

    var repeats = Math.max(1, opts.repeats || 1);
    var leadIn = Math.round((opts.leadIn == null ? 0.25 : opts.leadIn) * sampleRate);
    var leadOut = Math.round((opts.leadOut == null ? 0.25 : opts.leadOut) * sampleRate);
    var gap = Math.round((opts.gap == null ? 0.25 : opts.gap) * sampleRate);

    var totalLen = leadIn + leadOut + repeats * burst.length + (repeats - 1) * gap;
    var audio = new Float32Array(totalLen);
    var pos = leadIn;
    var syncOffsets = [];
    for (var r = 0; r < repeats; r++) {
      if (r > 0) pos += gap;
      syncOffsets.push(pos / sampleRate);
      audio.set(burst, pos);
      pos += burst.length;
    }

    var level = Math.pow(10, (opts.levelDbfs == null ? -3 : opts.levelDbfs) / 20);
    var peak = 0;
    for (var m = 0; m < audio.length; m++) peak = Math.max(peak, Math.abs(audio[m]));
    if (peak > 0) {
      var g = level / peak;
      for (var q = 0; q < audio.length; q++) audio[q] *= g;
    }

    return {
      audio: audio,
      sampleRate: sampleRate,
      take: payload.take,
      profile: profile.name,
      layout: built.layout,
      repeats: repeats,
      syncOffsets: syncOffsets,
      duration: totalLen / sampleRate
    };
  }

  function estimateDuration(opts) {
    opts = opts || {};
    var profile = PROFILES[opts.profile || 'fast'];
    if (!profile) throw new Error('unknown profile ' + opts.profile);
    var count = profile.preamble + profile.sfd + blockSymbols(PAYLOAD_BYTES, profile.sf);
    return { duration: count * symbolTime(profile, profile.sf) };
  }

  function toWavBlob(audio, sampleRate) {
    var n = audio.length;
    var buffer = new ArrayBuffer(44 + n * 2);
    var view = new DataView(buffer);
    function str(off, s) { for (var i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); }
    str(0, 'RIFF'); view.setUint32(4, 36 + n * 2, true); str(8, 'WAVE');
    str(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
    view.setUint16(22, 1, true); view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true);
    view.setUint16(34, 16, true); str(36, 'data'); view.setUint32(40, n * 2, true);
    for (var i = 0; i < n; i++) {
      var v = Math.max(-1, Math.min(1, audio[i]));
      view.setInt16(44 + i * 2, Math.round(v * 32767), true);
    }
    return buffer;
  }

  var api = {
    PROFILES: PROFILES,
    generate: generate,
    estimateDuration: estimateDuration,
    toWavBlob: toWavBlob,
    crc16: crc16,
    encodeBlock: encodeBlock,
    whiten: whiten,
    convEncode: convEncode
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.ChirpSync = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
