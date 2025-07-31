#ifndef SPECTRUMCONSTANTS_H
#define SPECTRUMCONSTANTS_H

#include <cstdint>

/*!
 * \brief Spectrum Instrumentation M4i series constants verified against SDK headers
 * 
 * This file defines constants from the Spectrum SDK headers (spcm/regs.h, spcm/spcerr.h)
 * specific to the M4i series of digitizers/AWGs. All values have been verified against 
 * the actual SDK headers to ensure accuracy.
 */

namespace Spectrum::M4i {

// ============================================================================
// Error Codes (from spcm/spcerr.h) - VERIFIED
// ============================================================================

constexpr std::int32_t ERR_OK                  = 0x0000;     /*!< No error */
constexpr std::int32_t ERR_INIT                = 0x0001;     /*!< Initialization error */
constexpr std::int32_t ERR_NR                  = 0x0002;     /*!< Board number error */
constexpr std::int32_t ERR_TYP                 = 0x0003;     /*!< Board type error */
constexpr std::int32_t ERR_FNCNOTSUPPORTED     = 0x0004;     /*!< Function not supported */
constexpr std::int32_t ERR_BRDREMAP            = 0x0005;     /*!< Board remapping error */
constexpr std::int32_t ERR_KERNELVERSION       = 0x0006;     /*!< Kernel version error */
constexpr std::int32_t ERR_HWDRVVERSION        = 0x0007;     /*!< Hardware driver version error */
constexpr std::int32_t ERR_ADRRANGE            = 0x0008;     /*!< Address range error */
constexpr std::int32_t ERR_INVALIDHANDLE       = 0x0009;     /*!< Invalid handle */

// ============================================================================
// Card Types (from spcm/regs.h) - VERIFIED
// ============================================================================

constexpr std::int32_t TYP_M4I2220_X8          = 0x00072220; /*!< M4i.2220-x8 FTMW digitizer */
constexpr std::int32_t TYP_M4I2211_X8          = 0x00072211; /*!< M4i.2211-x8 LIF digitizer */
constexpr std::int32_t TYP_M4I2230_X8          = 0x00072230; /*!< M4i.2230-x8 digitizer */

// ============================================================================
// Register Definitions (from spcm/regs.h) - VERIFIED
// ============================================================================

// Card Information Registers - VERIFIED  
constexpr std::int32_t SPC_PCITYP              = 2000;       /*!< Card type */
constexpr std::int32_t SPC_PCISERIALNO         = 2030;       /*!< Serial number */
constexpr std::int32_t SPC_PCIVERSION          = 2010;       /*!< Hardware version */
constexpr std::int32_t SPC_PCIDATE             = 2020;       /*!< Production date */

// Driver Information Registers - VERIFIED
constexpr std::int32_t SPC_GETDRVVERSION       = 1200;       /*!< Get driver version */
constexpr std::int32_t SPC_GETKERNELVERSION    = 1210;       /*!< Get kernel version */

// Memory Configuration - VERIFIED
constexpr std::int32_t SPC_MEMSIZE             = 10000;      /*!< Memory size in samples */
constexpr std::int32_t SPC_SEGMENTSIZE         = 10010;      /*!< Segment size */
constexpr std::int32_t SPC_LOOPS               = 10020;      /*!< Number of loops */
constexpr std::int32_t SPC_AVERAGES            = 10050;      /*!< Number of averages */
constexpr std::int32_t SPC_POSTTRIGGER         = 10100;      /*!< Post trigger size */

// Channel Configuration - VERIFIED
constexpr std::int32_t SPC_CHENABLE            = 11000;      /*!< Channel enable register */

// Channel Enable Flags - VERIFIED
constexpr std::int32_t CHANNEL0                = 0x00000001; /*!< Channel 0 enable */
constexpr std::int32_t CHANNEL1                = 0x00000002; /*!< Channel 1 enable */

// Card Mode - VERIFIED
constexpr std::int32_t SPC_CARDMODE            = 9500;       /*!< Card mode register */
constexpr std::int32_t SPC_REC_STD_SINGLE      = 0x00000001; /*!< Single shot recording */
constexpr std::int32_t SPC_REC_FIFO_MULTI      = 0x00000020; /*!< FIFO multiple recording */
constexpr std::int32_t SPC_REC_FIFO_AVERAGE    = 0x00200000; /*!< FIFO averaging mode */
constexpr std::int32_t SPC_REC_FIFO_AVERAGE_16BIT = 0x00400000; /*!< FIFO 16-bit averaging mode */

// Clock Configuration - VERIFIED
constexpr std::int32_t SPC_SAMPLERATE          = 20000;      /*!< Sample rate in Hz */
constexpr std::int32_t SPC_REFERENCECLOCK      = 20140;      /*!< Reference clock */
constexpr std::int32_t SPC_CLOCKMODE           = 20200;      /*!< Clock mode register */
constexpr std::int32_t SPC_CM_INTPLL           = 0x00000001; /*!< Internal PLL */
constexpr std::int32_t SPC_CM_EXTREFCLOCK      = 0x00000020; /*!< External reference clock */

// Channel-specific Registers - VERIFIED
constexpr std::int32_t SPC_AMP0                = 30010;      /*!< Channel 0 amplitude */
constexpr std::int32_t SPC_OFFS0               = 30000;      /*!< Channel 0 offset */
constexpr std::int32_t SPC_ACDC0               = 30020;      /*!< Channel 0 AC/DC coupling */
constexpr std::int32_t SPC_AMP1                = 30110;      /*!< Channel 1 amplitude */
constexpr std::int32_t SPC_OFFS1               = 30100;      /*!< Channel 1 offset */
constexpr std::int32_t SPC_ACDC1               = 30120;      /*!< Channel 1 AC/DC coupling */

// Trigger Configuration - VERIFIED
constexpr std::int32_t SPC_TRIG_ORMASK         = 40410;      /*!< Trigger OR mask */
constexpr std::int32_t SPC_TMASK_NONE          = 0x00000000; /*!< No trigger mask */
constexpr std::int32_t SPC_TMASK_EXT0          = 0x00000002; /*!< External trigger 0 mask */
constexpr std::int32_t SPC_TRIG_EXT0_MODE      = 40510;      /*!< External trigger 0 mode */
constexpr std::int32_t SPC_TRIG_EXT0_LEVEL0    = 42320;      /*!< External trigger 0 level 0 */
constexpr std::int32_t SPC_TRIG_DELAY          = 40810;      /*!< Trigger delay */
constexpr std::int32_t SPC_TM_POS              = 0x00000001; /*!< Positive edge trigger */
constexpr std::int32_t SPC_TM_NEG              = 0x00000002; /*!< Negative edge trigger */

// Status and Command Registers - VERIFIED
constexpr std::int32_t SPC_M2CMD               = 100;        /*!< Command register */
constexpr std::int32_t SPC_M2STATUS            = 110;        /*!< Status register */

// Commands - VERIFIED  
constexpr std::int32_t M2CMD_CARD_RESET        = 0x00000001; /*!< Hardware reset */
constexpr std::int32_t M2CMD_CARD_WRITESETUP   = 0x00000002; /*!< Write setup only */
constexpr std::int32_t M2CMD_CARD_START        = 0x00000004; /*!< Start card */
constexpr std::int32_t M2CMD_CARD_ENABLETRIGGER = 0x00000008; /*!< Enable trigger engine */
constexpr std::int32_t M2CMD_CARD_STOP         = 0x00000040; /*!< Stop run */
constexpr std::int32_t M2CMD_DATA_STARTDMA     = 0x00010000; /*!< Start DMA transfer */
constexpr std::int32_t M2CMD_DATA_WAITDMA      = 0x00020000; /*!< Wait for DMA transfer end */
constexpr std::int32_t M2CMD_DATA_STOPDMA      = 0x00040000; /*!< Stop DMA transfer */

// Status Bits - VERIFIED
constexpr std::int32_t M2STAT_CARD_READY       = 0x00000004; /*!< Card is ready, run finished */
constexpr std::int32_t M2STAT_DATA_BLOCKREADY  = 0x00000100; /*!< Next data block available */
constexpr std::int32_t M2STAT_DATA_END         = 0x00000200; /*!< Data transfer has ended */
constexpr std::int32_t M2STAT_DATA_ERROR       = 0x00000800; /*!< Internal error */

// Data Buffer Control Registers - VERIFIED
constexpr std::int32_t SPC_DATA_AVAIL_USER_LEN = 200;        /*!< Bytes available for user */
constexpr std::int32_t SPC_DATA_AVAIL_USER_POS = 201;        /*!< Current byte position of user data */
constexpr std::int32_t SPC_DATA_AVAIL_CARD_LEN = 202;        /*!< Bytes available for card */

// Transfer Direction Constants (from spcm_drv.h) - VERIFIED
constexpr std::int32_t SPCM_DIR_PCTOCARD       = 0;          /*!< PC to card transfer */
constexpr std::int32_t SPCM_DIR_CARDTOPC       = 1;          /*!< Card to PC transfer */

// Data Buffer Types (from spcm_drv.h) - VERIFIED
constexpr std::int32_t SPCM_BUF_DATA           = 1000;       /*!< Main data buffer */
constexpr std::int32_t SPCM_BUF_ABA            = 2000;       /*!< ABA data buffer */
constexpr std::int32_t SPCM_BUF_TIMESTAMP      = 3000;       /*!< Timestamp buffer */

} // namespace Spectrum::M4i

#endif // SPECTRUMCONSTANTS_H