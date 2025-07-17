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
    static const QString numConvolutionPoints{"catalogNumConvolutionPoints"};  // Number of points in convolution grid
    static const QString transitionCount{"catalogTransitionCount"};
    static const QString frequencyRange{"catalogFrequencyRange"};
    static const QString filterMinFreq{"catalogFilterMinFreq"};          // MHz - filtering range minimum
    static const QString filterMaxFreq{"catalogFilterMaxFreq"};          // MHz - filtering range maximum
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
    
    // Get FT data (for settings context)
    Ft getFtData() const;

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
    
    int numConvolutionPoints() const;      // Number of points in convolution grid
    void setNumConvolutionPoints(int numPoints);
    
    double calculatePointSpacing() const;   // Calculate spacing from range and number of points
    
    // Filtering range settings
    double filterMinFreq() const;          // MHz
    double filterMaxFreq() const;          // MHz  
    void setFilterRange(double minFreq, double maxFreq);
    
    // Convenience method to set all convolution parameters
    void setConvolutionSettings(bool enabled, LineshapeType lineshape, 
                               double linewidth, double minFreq, double maxFreq, 
                               int numPoints);
                               
    // Generate convolved spectrum from catalog data
    QVector<QPointF> generateConvolvedSpectrum() const;

    // Cache state management for background operations
    void invalidateConvolutionCache();
    void setCachePending();
    void setCacheValid(const QVector<QPointF> &convolvedData);
    bool isCacheValid() const;
    bool hasConvolvedData() const;

protected:
    void readFromDest() override;
    void writeToDest() override;
    void _storeMetadata(std::map<QString, QVariant> &m) override;
    void _retrieveMetadata(const std::map<QString, QVariant> &m) override;

private:
    QVector<QPointF> _xyData() const override;
    
    
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
    int d_numConvolutionPoints{1000};     // Number of points in convolution grid
    
    // Filtering range settings  
    double d_filterMinFreq{0.0};          // MHz
    double d_filterMaxFreq{1000.0};       // MHz
    
    // Cache state management
    enum class CacheState {
        Invalid,     // Cache is invalid/empty
        Pending,     // Background operation is in progress  
        Valid        // Cache contains valid data
    };
    
    // Cached convolved data
    mutable QVector<QPointF> d_convolvedCache;
    mutable CacheState d_cacheState{CacheState::Invalid};
};

#endif // OVERLAYTYPES_H

