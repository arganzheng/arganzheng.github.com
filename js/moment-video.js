(function (root) {
  var modulePromise;

  function loadMediabunny(base) {
    if (!modulePromise) {
      modulePromise = import((base || '').replace(/\/$/, '') + '/js/vendor/mediabunny-1.58.1.min.js');
    }
    return modulePromise;
  }

  function compress(file, options) {
    options = options || {};
    var cancelled = false, conversion = null, resolveCancelled;
    var cancelledPromise = new Promise(function (resolve) { resolveCancelled = resolve; });

    function cancel() {
      if (cancelled) return;
      cancelled = true;
      resolveCancelled(null);
      if (conversion) conversion.cancel().catch(function () {});
    }

    var work = (async function () {
      var input = null;
      try {
        if (typeof VideoEncoder === 'undefined') return null;
        var M = await loadMediabunny(options.base);
        if (cancelled) return null;

        input = new M.Input({ source: new M.BlobSource(file), formats: M.ALL_FORMATS });
        var duration = await input.computeDuration();
        if (cancelled) return null;
        var track = await input.getPrimaryVideoTrack();
        if (!track || !(await track.canDecode())) return null;
        if (cancelled) return null;
        var codec = await track.getCodecParameterString();
        if (cancelled) return null;

        var displayWidth = track.displayWidth, displayHeight = track.displayHeight;
        var shortSide = Math.min(displayWidth, displayHeight);
        if (codec && codec.indexOf('avc1') === 0 && shortSide <= 720 && duration > 0 &&
            file.size * 8 / duration <= 2500000) return null;

        var scale = Math.min(1, 720 / shortSide);
        var width = Math.round(displayWidth * scale / 2) * 2;
        var height = Math.round(displayHeight * scale / 2) * 2;
        var output = new M.Output({
          format: new M.Mp4OutputFormat({ fastStart: 'in-memory' }),
          target: new M.BufferTarget()
        });
        conversion = await M.Conversion.init({
          input: input,
          output: output,
          video: { width: width, height: height, fit: 'fill', codec: 'avc', bitrate: 1200000, forceTranscode: true },
          audio: { codec: 'aac', bitrate: 96000 }
        });
        if (cancelled) {
          conversion.cancel().catch(function () {});
          return null;
        }
        if (!conversion.isValid || conversion.discardedTracks.some(function (item) { return item.track.type === 'video'; })) {
          conversion.cancel().catch(function () {});
          return null;
        }
        conversion.onProgress = function (progress) {
          if (cancelled || typeof options.onProgress !== 'function') return;
          try { options.onProgress(progress); } catch (e) { /* ignore */ }
        };
        await conversion.execute();
        if (cancelled) return null;
        var buffer = output.target.buffer;
        if (!buffer || buffer.byteLength >= file.size) return null;
        var name = file.name || 'video';
        var basename = name.replace(/\.[^.]*$/, '') || 'video';
        return new File([buffer], basename + '.mp4', { type: 'video/mp4' });
      } catch (e) {
        return null;
      } finally {
        if (input) { try { input.dispose(); } catch (e) { /* ignore */ } }
      }
    })();

    return { promise: Promise.race([work, cancelledPromise]), cancel: cancel };
  }

  root.MomentVideo = { compress: compress };
})(window);
