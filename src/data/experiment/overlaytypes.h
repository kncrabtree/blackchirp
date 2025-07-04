#ifndef OVERLAYTYPES_H
#define OVERLAYTYPES_H

#include <data/experiment/overlaybase.h>
#include <data/analysis/ftworker.h>
#include <data/analysis/ft.h>

namespace BC::Key::Overlay {
// FT metadata keys
static const QString ftYMin{"ftYMin"};
static const QString ftYMax{"ftYMax"};
static const QString ftX0MHz{"ftX0MHz"};
static const QString ftSpacingMHz{"ftSpacingMHz"};
static const QString ftLoFreqMHz{"ftLoFreqMHz"};
static const QString ftShots{"ftShots"};

// BCExperiment overlay specific keys
static const QString frame{"frame"};
static const QString procStartUs{"procStartUs"};
static const QString procEndUs{"procEndUs"};
static const QString procExpFilter{"procExpFilter"};
static const QString procZeroPadFactor{"procZeroPadFactor"};
static const QString procRemoveDC{"procRemoveDC"};
static const QString procUnits{"procUnits"};
static const QString procAutoScaleIgnoreMHz{"procAutoScaleIgnoreMHz"};
static const QString procWindowFunction{"procWindowFunction"};
}

class BCExpOverlay : public OverlayBase
{
public:
    BCExpOverlay(const Ft &ft, int frame = -1);

    // OverlayBase interface
    QVector<QPointF> xyData() const override;

    // Setters/getters for processing settings (for user override)
    void setProcessingSettings(const FtWorker::FidProcessingSettings &settings) { d_processingSettings = settings; }
    FtWorker::FidProcessingSettings getProcessingSettings() const { return d_processingSettings; }

protected:
    void readFromSource() override;
    void readFromDest() override;
    void writeToDest() override;
    void _storeMetadata(std::map<QString, QVariant> &m) override;
    void _retrieveMetadata(const std::map<QString, QVariant> &m) override;


private:
    Ft d_ft;
    int d_frame{-1};  // Frame to process (-1 = averaged data)
    FtWorker::FidProcessingSettings d_processingSettings;

};

#endif // OVERLAYTYPES_H

