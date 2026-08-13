#include <QtTest>
#include <QElapsedTimer>
#include <QThread>

#include <atomic>

#include <data/experiment/overlaytypes.h>

/*!
 * \brief Concurrency coverage for CatalogOverlay's shared cache/data state.
 *
 * In production, ConvolutionOperation::execute() runs on a QtConcurrent
 * worker thread and mutates a live CatalogOverlay that the GUI thread is
 * simultaneously reading (FtPlot::addOverlay()/updateOverlay() calling
 * xyData()/displayYRange()) and writing (CatalogOverlayWidget::applyToOverlay()
 * reacting to real-time spinbox edits). This test reproduces that shape with
 * one thread repeatedly running the convolution sequence a real
 * ConvolutionOperation performs (setCachePending() -> generateConvolvedSpectrum()
 * -> setCacheValid()) while another thread hammers the read/write surface
 * used by the GUI thread. Before the fix, the unsynchronized QVector and
 * CatalogData members made this a use-after-free; after it, every access is
 * serialized through OverlayBase::d_mutex and the convolution loop itself
 * runs without the lock held.
 */
class CatalogOverlayThreadSafetyTest : public QObject
{
    Q_OBJECT

private slots:
    void convolutionRacesReadsAndWrites();

private:
    static CatalogData makeCatalogData(int numTransitions);
};

CatalogData CatalogOverlayThreadSafetyTest::makeCatalogData(int numTransitions)
{
    QVector<TransitionData> transitions;
    transitions.reserve(numTransitions);
    for (int i = 0; i < numTransitions; ++i) {
        double freq = 1000.0 + 8000.0 * i / static_cast<double>(numTransitions);
        double intensity = 1.0 + (i % 7);
        transitions.append(TransitionData(freq, intensity, QString("J=%1").arg(i)));
    }

    CatalogData data;
    data.setTransitions(transitions);
    return data;
}

void CatalogOverlayThreadSafetyTest::convolutionRacesReadsAndWrites()
{
    auto overlay = std::make_shared<CatalogOverlay>();
    overlay->setCatalogData(makeCatalogData(400));
    overlay->setConvolutionSettings(true, CatalogOverlay::Lorentzian,
                                     50.0, 1000.0, 9000.0, 20000);

    // Bounds on both threads so a regression (deadlock or runaway loop)
    // cannot hang CI: a hard wall-clock budget plus an iteration cap.
    const int runMs = 1500;
    const int maxIterations = 100000;

    std::atomic<bool> stop{false};
    std::atomic<int> convolutionsCompleted{0};
    std::atomic<int> readerIterations{0};

    // Worker thread: repeats the exact sequence ConvolutionOperation::execute()
    // performs on a QtConcurrent thread in production.
    QThread *worker = QThread::create([&]() {
        QElapsedTimer timer;
        timer.start();
        int iterations = 0;
        while (!stop.load(std::memory_order_relaxed)
               && timer.elapsed() < runMs
               && iterations < maxIterations) {
            overlay->setCachePending();
            auto convolved = overlay->generateConvolvedSpectrum();
            overlay->setCacheValid(convolved);
            convolutionsCompleted.fetch_add(1, std::memory_order_relaxed);
            ++iterations;
        }
    });

    // Reader/writer thread: mirrors the GUI-thread access pattern -- plot
    // reads (xyData(), displayYRange(), yMax()) interleaved with the live
    // settings edits CatalogOverlayWidget::applyToOverlay() makes.
    QThread *reader = QThread::create([&]() {
        QElapsedTimer timer;
        timer.start();
        int iterations = 0;
        double minFreq = 1000.0, maxFreq = 9000.0;
        while (!stop.load(std::memory_order_relaxed)
               && timer.elapsed() < runMs
               && iterations < maxIterations) {
            const auto data = overlay->xyData();
            // Self-consistency: every point returned must be a real,
            // finite number -- a torn/partially-overwritten QVector under
            // concurrent access would produce garbage or crash outright.
            for (const QPointF &p : data) {
                if (!std::isfinite(p.x()) || !std::isfinite(p.y())) {
                    QFAIL("xyData() returned a non-finite point under concurrent convolution");
                }
            }

            const auto range = overlay->displayYRange();
            Q_UNUSED(range);
            overlay->yMax();

            overlay->isCacheValid();
            overlay->hasConvolvedData();
            overlay->catalogData();

            // Perturb settings slightly each pass, like a spinbox edit
            // arriving mid-convolution.
            minFreq = 1000.0 + (iterations % 50);
            maxFreq = 9000.0 - (iterations % 50);
            overlay->setConvolutionFreqRange(minFreq, maxFreq);
            overlay->setLinewidth(40.0 + (iterations % 20));

            ++iterations;
        }
        readerIterations.store(iterations, std::memory_order_relaxed);
    });

    worker->start();
    reader->start();

    // Generous timeouts relative to runMs so normal scheduling jitter can't
    // trip the bound, while still guaranteeing termination.
    QVERIFY(worker->wait(runMs + 10000));
    stop.store(true, std::memory_order_relaxed);
    QVERIFY(reader->wait(runMs + 10000));

    delete worker;
    delete reader;

    // Reaching here without a crash, hang, or QFAIL is the primary assertion.
    QVERIFY(convolutionsCompleted.load() > 0);
    QVERIFY(readerIterations.load() > 0);

    // Final state must be self-consistent regardless of which thread's
    // writes landed last.
    const auto finalData = overlay->xyData();
    for (const QPointF &p : finalData) {
        QVERIFY(std::isfinite(p.x()));
        QVERIFY(std::isfinite(p.y()));
    }
}

QTEST_MAIN(CatalogOverlayThreadSafetyTest)
#include "tst_catalogoverlaythreadsafety.moc"
