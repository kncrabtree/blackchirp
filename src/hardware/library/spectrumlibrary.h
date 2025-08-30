#ifndef SPECTRUMLIBRARY_H  
#define SPECTRUMLIBRARY_H

#include "vendorlibrary.h"
#include <cstdint>

namespace BC::Key::Spectrum {
    static const QString spectrumM4i{"spectrumM4i"};  /*!< Settings key for Spectrum M4i library */
}

/*!
 * \brief Dynamic loader for Spectrum Instrumentation driver library
 * 
 * This class provides runtime loading of the Spectrum Instrumentation driver
 * library (spcm_linux) without requiring it at compile time. This allows
 * BlackChirp to be built and distributed without the Spectrum SDK, while
 * still supporting Spectrum hardware when the driver is installed.
 * 
 * The class uses the singleton pattern to ensure only one instance exists,
 * as the Spectrum library maintains global state that shouldn't be duplicated.
 * 
 * Usage:
 * ```cpp
 * SpectrumLibrary& lib = SpectrumLibrary::instance();
 * if (lib.isAvailable()) {
 *     void* handle = lib.spcm_hOpen("/dev/spcm0");
 *     // ... use other Spectrum functions
 * }
 * ```
 * 
 * The library automatically searches common installation paths:
 * - System library paths (LD_LIBRARY_PATH)
 * - /opt/spectrum/lib (default Spectrum installation)
 * - /usr/local/lib (manual installation)
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
    bool isAvailable() const override { return d_libraryLoaded; }
    QString errorString() const override { return d_errorString; }
    QString libraryName() const override { return "spcm"; }
    QStringList platformLibraryNames() const override;
    QStringList defaultSearchPaths() const override;
    QString getVersionInfo() const override;

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