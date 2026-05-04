#ifndef SPECTRUMLIBRARY_H  
#define SPECTRUMLIBRARY_H

#include "vendorlibrary.h"
#include <cstdint>

namespace BC::Key::Spectrum {
    inline constexpr QLatin1StringView spectrumM4i{"spectrumM4i"};  /*!< Settings key for Spectrum M4i library */
}

/*!
 * \brief Dynamic loader for the Spectrum Instrumentation driver
 * library (\c spcm_linux / \c spcm64.dll).
 *
 * Singleton, because the Spectrum library maintains global state that
 * cannot be duplicated. Used by the M4i digitizer implementations.
 */
class SpectrumLibrary : public VendorLibrary
{
    Q_OBJECT

public:
    /*!
     * \brief Get singleton instance of SpectrumLibrary
     * \return Reference to the single SpectrumLibrary instance
     */
    static SpectrumLibrary& instance();

    // VendorLibrary interface
    /*!
     * \brief Check if the Spectrum driver library is loaded and ready
     * \return true if library symbols were resolved successfully
     */
    bool isAvailable() const override { return d_libraryLoaded; }
    /*!
     * \brief Get error message if library loading failed
     * \return Error string describing why the library is not available
     */
    QString errorString() const override { return d_errorString; }
    /*!
     * \brief Get the base name of the Spectrum driver library
     * \return Base library name string ("spcm")
     */
    QString libraryName() const override { return "spcm"; }
    /*!
     * \brief Get platform-specific library file names to try when loading
     * \return List of library names with platform-specific variations
     */
    QStringList platformLibraryNames() const override;
    /*!
     * \brief Get default search paths for automatic Spectrum driver discovery
     * \return List of default directories to search for the library
     */
    QStringList defaultSearchPaths() const override;
    /*!
     * \brief Get Spectrum driver version string
     * \return Version string from the loaded driver, or empty string if unavailable
     */
    QString getVersionInfo() const override;
    /*!
     * \brief Get platform-specific installation instructions for the Spectrum driver
     * \return HTML-formatted instructions for installing the Spectrum driver on the current platform
     */
    QString getInstallationInstructions() const override;

    // Spectrum API function pointers
    // These replace direct SDK function calls and are loaded dynamically
    
    /*!
     * \brief Open Spectrum device
     * Function pointer for spcm_hOpen()
     */
    typedef void* (*spcm_hOpen_t)(const char* szDeviceName);
    spcm_hOpen_t spcm_hOpen = nullptr;

    /*!
     * \brief Close Spectrum device  
     * Function pointer for spcm_vClose()
     */
    typedef void (*spcm_vClose_t)(void* hDevice);
    spcm_vClose_t spcm_vClose = nullptr;

    /*!
     * \brief Set 32-bit integer parameter
     * Function pointer for spcm_dwSetParam_i32()
     */
    typedef std::int32_t (*spcm_dwSetParam_i32_t)(void* hDevice, std::int32_t lRegister, std::int32_t lValue);
    spcm_dwSetParam_i32_t spcm_dwSetParam_i32 = nullptr;

    /*!
     * \brief Get 32-bit integer parameter
     * Function pointer for spcm_dwGetParam_i32()
     */
    typedef std::int32_t (*spcm_dwGetParam_i32_t)(void* hDevice, std::int32_t lRegister, std::int32_t* plValue);
    spcm_dwGetParam_i32_t spcm_dwGetParam_i32 = nullptr;

    /*!
     * \brief Set 64-bit integer parameter
     * Function pointer for spcm_dwSetParam_i64()
     */
    typedef std::int32_t (*spcm_dwSetParam_i64_t)(void* hDevice, std::int32_t lRegister, std::int64_t llValue);
    spcm_dwSetParam_i64_t spcm_dwSetParam_i64 = nullptr;

    /*!
     * \brief Get 64-bit integer parameter
     * Function pointer for spcm_dwGetParam_i64()
     */
    typedef std::int32_t (*spcm_dwGetParam_i64_t)(void* hDevice, std::int32_t lRegister, std::int64_t* pllValue);
    spcm_dwGetParam_i64_t spcm_dwGetParam_i64 = nullptr;

    /*!
     * \brief Define transfer buffer
     * Function pointer for spcm_dwDefTransfer_i64()
     */
    typedef std::int32_t (*spcm_dwDefTransfer_i64_t)(void* hDevice, std::int32_t lBufType, 
                                                     std::int32_t lDirection, std::int32_t lNotifySize,
                                                     void* pvDataBuffer, std::int64_t qwOffset,
                                                     std::int64_t qwLength);
    spcm_dwDefTransfer_i64_t spcm_dwDefTransfer_i64 = nullptr;

    /*!
     * \brief Start transfer
     * Function pointer for spcm_dwInvalidateBuf()  
     */
    typedef std::int32_t (*spcm_dwInvalidateBuf_t)(void* hDevice, std::int32_t lBufType);
    spcm_dwInvalidateBuf_t spcm_dwInvalidateBuf = nullptr;

    /*!
     * \brief Get error information
     * Function pointer for spcm_dwGetErrorInfo_i32()
     */
    typedef std::int32_t (*spcm_dwGetErrorInfo_i32_t)(void* hDevice, std::int32_t* plErrorReg, 
                                                      std::int32_t* plErrorValue, char* szErrorText);
    spcm_dwGetErrorInfo_i32_t spcm_dwGetErrorInfo_i32 = nullptr;

protected:
    /*!
     * \brief Resolve Spectrum driver function symbols after the library has loaded
     */
    void loadFunctions() override;

private:
    explicit SpectrumLibrary(QObject *parent = nullptr);
    ~SpectrumLibrary() = default;
    
    // Singleton - disable copy/assignment
    SpectrumLibrary(const SpectrumLibrary&) = delete;
    SpectrumLibrary& operator=(const SpectrumLibrary&) = delete;

    static SpectrumLibrary* s_instance;
};

#endif // SPECTRUMLIBRARY_H