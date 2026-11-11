export default function({parentElement, data}) {
    // v2 invokes this renderer on data changes without replacing parentElement.
    let plot = parentElement.querySelector('.training-live-plot');
    if (!plot) {
        plot = document.createElement('div');
        plot.className = 'training-live-plot';
        plot.style.width = '100%';
        parentElement.appendChild(plot);
    }
    if (plot._spec !== data) {
        plot._spec = data;
        const figure = JSON.parse(data);
        // Serialize updates: a slow draw must not race the next live snapshot.
        plot.pending = (plot.pending || Promise.resolve()).then(async () => {
            if (!window.Plotly) {
                await new Promise((resolve, reject) => {
                    const ready = () => {
                        if (!window.Plotly) return;
                        clearTimeout(timer);
                        window.removeEventListener('klab-plotly-ready', ready);
                        resolve();
                    };
                    const timer = setTimeout(() => {
                        window.removeEventListener('klab-plotly-ready', ready);
                        reject(new Error('Plotly loader did not become ready within 15 seconds'));
                    }, 15_000);
                    window.addEventListener('klab-plotly-ready', ready);
                });
            }
            if (!plot.isConnected) return;
            if (plot.layout?.uirevision === figure.layout.uirevision) {
                const previous = new Map(plot.data.map(trace => [trace.uid, trace]));
                for (const trace of figure.data) {
                    const old = previous.get(trace.uid);
                    for (const field of ['visible', 'selectedpoints']) {
                        if (old && Object.hasOwn(old, field)) trace[field] = old[field];
                    }
                }
            }
            return window.Plotly.react(plot, figure.data, figure.layout, {
                responsive: false, displaylogo: false, scrollZoom: false
            });
        }).catch(error => {
            console.error('Training chart update failed', error);
            if (plot._spec === data) delete plot._spec;
            let message = parentElement.querySelector('[role="alert"]');
            if (!message) {
                message = document.createElement('p');
                message.setAttribute('role', 'alert');
                parentElement.appendChild(message);
            }
            message.textContent = 'Chart update failed; waiting for the next data update.';
        });
        plot.pending.then(() => {
            if (plot._spec === data) parentElement.querySelector('[role="alert"]')?.remove();
        });
    }
    const observer = new ResizeObserver(() => {
        if (plot._fullLayout) window.Plotly.Plots.resize(plot);
    });
    observer.observe(parentElement);
    // Cleanup runs before the next render too: retain the plot and its UI state.
    return () => {
        observer.disconnect();
        queueMicrotask(() => {
            if (!plot.isConnected) {
                plot.pending?.finally(() => window.Plotly?.purge(plot));
            }
        });
    };
}
