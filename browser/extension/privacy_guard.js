(() => {
  const own = Object.prototype.hasOwnProperty;
  const hideFunction = (target, key) => {
    if (!target) return;
    try {
      Object.defineProperty(target, key, {
        configurable: true,
        enumerable: false,
        get: () => undefined,
        set: () => undefined
      });
    } catch (_) {}
  };
  const removeFromChain = (target, key) => {
    let current = target;
    while (current) {
      if (own.call(current, key)) {
        try { delete current[key]; } catch (_) {}
      }
      current = Object.getPrototypeOf(current);
    }
  };
  const setTrue = (target, key) => {
    if (!target) return;
    try {
      Object.defineProperty(target, key, {
        configurable: true,
        enumerable: false,
        get: () => true
      });
    } catch (_) {}
  };

  for (const key of ["browsingTopics", "hasPrivateToken"]) hideFunction(document, key);
  for (const key of ["joinAdInterestGroup", "leaveAdInterestGroup", "runAdAuction", "updateAdInterestGroups"])
    hideFunction(navigator, key);
  if (globalThis.HTMLAnchorElement) removeFromChain(HTMLAnchorElement.prototype, "attributionSrc");
  setTrue(navigator, "globalPrivacyControl");
  if (globalThis.Navigator) setTrue(Navigator.prototype, "globalPrivacyControl");

  let seed = 2166136261;
  for (const character of location.origin) {
    seed ^= character.charCodeAt(0);
    seed = Math.imul(seed, 16777619);
  }
  try {
    const random = new Uint32Array(1);
    crypto.getRandomValues(random);
    seed ^= random[0];
  } catch (_) {
    seed ^= Math.floor(Math.random() * 0xffffffff);
  }
  seed >>>= 0;

  const canvasPrototype = globalThis.CanvasRenderingContext2D?.prototype;
  const originalGetImageData = canvasPrototype?.getImageData;
  if (typeof originalGetImageData === "function") {
    Object.defineProperty(canvasPrototype, "getImageData", {
      configurable: true,
      writable: true,
      value: function getImageData(...args) {
        const image = originalGetImageData.apply(this, args);
        const channels = image?.data;
        if (channels?.length >= 4) {
          const pixelCount = Math.floor(channels.length / 4);
          const channel = ((seed % pixelCount) * 4) + ((seed >>> 8) % 3);
          channels[channel] = channels[channel] === 255 ? 254 : channels[channel] + 1;
        }
        return image;
      }
    });
  }

  const audioPrototype = globalThis.AudioBuffer?.prototype;
  const originalGetChannelData = audioPrototype?.getChannelData;
  const protectedChannels = new WeakMap();
  if (typeof originalGetChannelData === "function") {
    Object.defineProperty(audioPrototype, "getChannelData", {
      configurable: true,
      writable: true,
      value: function getChannelData(channelNumber) {
        const samples = originalGetChannelData.call(this, channelNumber);
        let channels = protectedChannels.get(this);
        if (!channels) {
          channels = new Set();
          protectedChannels.set(this, channels);
        }
        if (!channels.has(channelNumber) && samples.length) {
          const index = seed % samples.length;
          const delta = (seed & 1) === 0 ? 0.0000001 : -0.0000001;
          samples[index] = Math.max(-1, Math.min(1, samples[index] + delta));
          channels.add(channelNumber);
        }
        return samples;
      }
    });
  }
})();
