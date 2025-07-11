#ifndef OVERLAYTYPES_H
#define OVERLAYTYPES_H

#include <data/experiment/overlaybase.h>
#include <data/analysis/ft.h>
#include <data/experiment/catalogdata.h>

namespace BC::Key::Overlay {
// FT metadata keys
static const QString ftYMin{"ftYMin"};
static const QString ftYMax{"ftYMax"};
static const QString ftX0MHz{"ftX0MHz"};
static const QString ftSpacingMHz{"ftSpacingMHz"};
static const QString ftLoFreqMHz{"ftLoFreqMHz"};
static const QString ftShots{"ftShots"};

// Catalog overlay specific keys
namespace Catalog {
    static const QString sourceProgram{"catalogSourceProgram"};
    static const QString moleculeName{"catalogMoleculeName"};
    static const QString convolutionEnabled{"catalogConvolutionEnabled"};
    static const QString lineshapeType{"catalogLineshapeType"};
    static const QString linewidthKHz{"catalogLinewidthKHz"};            // FWHM in kHz
    static const QString convolutionMinFreq{"catalogConvolutionMinFreq"}; // MHz
    static const QString convolutionMaxFreq{"catalogConvolutionMaxFreq"}; // MHz
    static const QString pointSpacing{"catalogPointSpacing"};            // MHz
    static const QString transitionCount{"catalogTransitionCount"};
    static const QString frequencyRange{"catalogFrequencyRange"};
}
}

class BCExpOverlay : public OverlayBase
{
public:
    BCExpOverlay();

    // OverlayBase interface
private:
    QVector<QPointF> _xyData() const override;

public:

    // Set FT data directly (called after creation)
    void setFtData(const Ft &ftData);

protected:
    void readFromDest() override;
    void writeToDest() override;
    void _storeMetadata(std::map<QString, QVariant> &m) override;
    void _retrieveMetadata(const std::map<QString, QVariant> &m) override;


private:
    Ft d_ft;

};

/**
 * @brief Catalog overlay for spectroscopic transition data with convolution support
 * 
 * This class handles catalog data from various programs (SPCAT, XIAM, etc.) and
 * provides optional convolution with lineshape functions for realistic spectrum comparison.
 */
class CatalogOverlay : public OverlayBase
{
public:
    enum LineshapeType {
        Lorentzian,
        Gaussian
    };
    
    CatalogOverlay();
    
    // Catalog data management
    CatalogData catalogData() const;
    void setCatalogData(const CatalogData &data);
    
    // Convolution settings
    bool convolutionEnabled() const;
    void setConvolutionEnabled(bool enabled);
    
    LineshapeType lineshapeType() const;
    void setLineshapeType(LineshapeType type);
    
    double linewidth() const;              // FWHM in kHz
    void setLinewidth(double width);
    
    double convolutionMinFreq() const;     // MHz
    double convolutionMaxFreq() const;     // MHz
    void setConvolutionFreqRange(double minFreq, double maxFreq);
    
    double pointSpacing() const;           // MHz
    void setPointSpacing(double spacing);
    
    // Convenience method to set all convolution parameters
    void setConvolutionSettings(bool enabled, LineshapeType lineshape, 
                               double linewidth, double minFreq, double maxFreq, 
                               double spacing);

protected:
    void readFromDest() override;
    void writeToDest() override;
    void _storeMetadata(std::map<QString, QVariant> &m) override;
    void _retrieveMetadata(const std::map<QString, QVariant> &m) override;

private:
    QVector<QPointF> _xyData() const override;
    
    // Generate convolved spectrum from catalog data
    QVector<QPointF> generateConvolvedSpectrum() const;
    
    // Lineshape functions (x and x0 in MHz, width in kHz)
    double lorentzianProfile(double x, double x0, double fwhmKHz) const;
    double gaussianProfile(double x, double x0, double fwhmKHz) const;
    
    // Data members
    CatalogData d_catalogData;
    
    // Convolution settings
    bool d_convolutionEnabled{false};
    LineshapeType d_lineshapeType{Lorentzian};
    double d_linewidth{100.0};            // FWHM in kHz
    double d_convolutionMinFreq{0.0};     // MHz
    double d_convolutionMaxFreq{1000.0};  // MHz
    double d_pointSpacing{0.01};          // MHz
    
    // Cached convolved data
    mutable QVector<QPointF> d_convolvedCache;
    mutable bool d_convolutionCacheValid{false};
    
    void invalidateConvolutionCache();
};

#endif // OVERLAYTYPES_H

