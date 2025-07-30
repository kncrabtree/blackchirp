#ifndef SPECTRUMCONSTANTS_H
#define SPECTRUMCONSTANTS_H

#include <cstdint>

/*!
 * \brief Spectrum Instrumentation M4i series constants without requiring vendor headers
 * 
 * This file defines constants from the Spectrum SDK headers (spcm/regs.h, spcm/dlltyp.h)
 * specific to the M4i series of digitizers/AWGs. This eliminates compile-time dependencies 
 * on the vendor SDK. These constants are stable across SDK versions and allow BlackChirp 
 * to be built without the Spectrum SDK installed.
 * 
 * Constants are organized by functional area and include comprehensive documentation
 * for easier understanding without referring to vendor documentation.
 * 
 * Note: Other Spectrum card families (M2i, M3i, etc.) may have different constants
 * and would require separate constant definitions.
 */

namespace Spectrum::M4i {

// ============================================================================
// Error Codes (from spcm/spcerr.h)
// ============================================================================

constexpr std::int32_t ERR_OK                  = 0x00000000;  /*!< No error */
constexpr std::int32_t ERR_INIT                = 0x00000001;  /*!< Initialization error */
constexpr std::int32_t ERR_NR                  = 0x00000002;  /*!< Board number error */
constexpr std::int32_t ERR_TYP                 = 0x00000003;  /*!< Board type error */
constexpr std::int32_t ERR_FNCNOTSUPPORTED     = 0x00000004;  /*!< Function not supported */
constexpr std::int32_t ERR_BRDREMAP            = 0x00000005;  /*!< Board remapping error */
constexpr std::int32_t ERR_KERNELVERSION       = 0x00000006;  /*!< Kernel version error */
constexpr std::int32_t ERR_HWDRVVERSION        = 0x00000007;  /*!< Hardware driver version error */
constexpr std::int32_t ERR_ADRRANGE            = 0x00000008;  /*!< Address range error */
constexpr std::int32_t ERR_INVALIDHANDLE       = 0x00000009;  /*!< Invalid handle */

// ============================================================================
// Card Types (from spcm/dlltyp.h) 
// ============================================================================

constexpr std::int32_t TYP_M4I2220             = 0x22200000;  /*!< M4i.2220-x8 FTMW digitizer */
constexpr std::int32_t TYP_M4I2211             = 0x22110000;  /*!< M4i.2211-x8 LIF digitizer */
constexpr std::int32_t TYP_M4I2230             = 0x22300000;  /*!< M4i.2230-x4 digitizer */
constexpr std::int32_t TYP_M4I4420             = 0x44200000;  /*!< M4i.4420-x8 AWG */
constexpr std::int32_t TYP_M4I4450             = 0x44500000;  /*!< M4i.4450-x4 AWG */

// ============================================================================
// Register Definitions (from spcm/regs.h)
// ============================================================================

// Card Information Registers
constexpr std::int32_t SPC_PCITYP              = 0x00000000;  /*!< Card type */
constexpr std::int32_t SPC_PCISERIALNO         = 0x00000001;  /*!< Serial number */
constexpr std::int32_t SPC_PCIVERSION          = 0x00000002;  /*!< Hardware version */
constexpr std::int32_t SPC_PCIDATE             = 0x00000003;  /*!< Production date */
constexpr std::int32_t SPC_PCIDRVVERSION       = 0x00000004;  /*!< Driver version */
constexpr std::int32_t SPC_PCIKERNELVERSION    = 0x00000005;  /*!< Kernel version */
constexpr std::int32_t SPC_PCIBASEADR0         = 0x00000006;  /*!< Base address 0 */
constexpr std::int32_t SPC_PCIBASEADR1         = 0x00000007;  /*!< Base address 1 */

// Memory and Channel Configuration
constexpr std::int32_t SPC_CHENABLE            = 0x00000100;  /*!< Channel enable register */
constexpr std::int32_t SPC_MEMSIZE             = 0x00000101;  /*!< Memory size in samples */
constexpr std::int32_t SPC_SEGMENTSIZE         = 0x00000102;  /*!< Segment size */
constexpr std::int32_t SPC_POSTTRIGGER         = 0x00000103;  /*!< Post trigger size */
constexpr std::int32_t SPC_PRETRIGGER          = 0x00000104;  /*!< Pre trigger size */

// Acquisition Mode
constexpr std::int32_t SPC_CARDMODE            = 0x00000200;  /*!< Card mode register */
constexpr std::int32_t SPC_LOOPS               = 0x00000201;  /*!< Number of loops */
constexpr std::int32_t SPC_AVERAGES            = 0x00000202;  /*!< Number of averages */
constexpr std::int32_t SPC_SEGMENTCOUNT        = 0x00000203;  /*!< Number of segments */

// Card Modes
constexpr std::int32_t SPC_REC_STD_SINGLE      = 0x00000001;  /*!< Standard single shot */
constexpr std::int32_t SPC_REC_STD_MULTI       = 0x00000002;  /*!< Standard multiple recording */
constexpr std::int32_t SPC_REC_STD_GATE        = 0x00000004;  /*!< Standard gated sampling */
constexpr std::int32_t SPC_REC_STD_ABA         = 0x00000008;  /*!< Standard ABA mode */
constexpr std::int32_t SPC_REC_FIFO_SINGLE     = 0x00000010;  /*!< FIFO single shot */
constexpr std::int32_t SPC_REC_FIFO_MULTI      = 0x00000020;  /*!< FIFO multiple recording */

// Clock and Trigger
constexpr std::int32_t SPC_SAMPLERATE          = 0x00000300;  /*!< Sample rate in Hz */
constexpr std::int32_t SPC_EXTERNOUT           = 0x00000301;  /*!< External output */
constexpr std::int32_t SPC_CLOCKOUT            = 0x00000302;  /*!< Clock output */
constexpr std::int32_t SPC_CLOCK50OHM          = 0x00000303;  /*!< Clock 50 ohm termination */
constexpr std::int32_t SPC_REFERENCECLOCK      = 0x00000304;  /*!< Reference clock */

constexpr std::int32_t SPC_TRIG_EXT0_MODE      = 0x00000400;  /*!< External trigger 0 mode */
constexpr std::int32_t SPC_TRIG_EXT0_LEVEL0    = 0x00000401;  /*!< External trigger 0 level 0 */
constexpr std::int32_t SPC_TRIG_EXT0_LEVEL1    = 0x00000402;  /*!< External trigger 0 level 1 */
constexpr std::int32_t SPC_TRIG_EXT0_PULSEWIDTH = 0x00000403; /*!< External trigger 0 pulse width */

// Trigger Modes
constexpr std::int32_t SPC_TM_POS               = 0x00000001;  /*!< Positive edge trigger */
constexpr std::int32_t SPC_TM_NEG               = 0x00000002;  /*!< Negative edge trigger */
constexpr std::int32_t SPC_TM_BOTH              = 0x00000003;  /*!< Both edge trigger */
constexpr std::int32_t SPC_TM_HIGH              = 0x00000004;  /*!< High level trigger */
constexpr std::int32_t SPC_TM_LOW               = 0x00000005;  /*!< Low level trigger */

// Channel-specific registers (CH0, CH1, etc.)
constexpr std::int32_t SPC_AMP0                = 0x00000500;  /*!< Channel 0 amplitude */
constexpr std::int32_t SPC_AMP1                = 0x00000501;  /*!< Channel 1 amplitude */
constexpr std::int32_t SPC_OFFS0               = 0x00000510;  /*!< Channel 0 offset */
constexpr std::int32_t SPC_OFFS1               = 0x00000511;  /*!< Channel 1 offset */
constexpr std::int32_t SPC_ACDC0               = 0x00000520;  /*!< Channel 0 AC/DC coupling */
constexpr std::int32_t SPC_ACDC1               = 0x00000521;  /*!< Channel 1 AC/DC coupling */
constexpr std::int32_t SPC_50OHM0              = 0x00000530;  /*!< Channel 0 50 ohm termination */
constexpr std::int32_t SPC_50OHM1              = 0x00000531;  /*!< Channel 1 50 ohm termination */

// Coupling modes
constexpr std::int32_t COUPLING_DC             = 0x00000000;  /*!< DC coupling */
constexpr std::int32_t COUPLING_AC             = 0x00000001;  /*!< AC coupling */

// Status and Control
constexpr std::int32_t SPC_M2STATUS            = 0x00000600;  /*!< Status register */
constexpr std::int32_t SPC_M2CMD               = 0x00000601;  /*!< Command register */
constexpr std::int32_t SPC_TIMEOUT             = 0x00000602;  /*!< Timeout value */

// Commands
constexpr std::int32_t M2CMD_CARD_START        = 0x00000001;  /*!< Start card */
constexpr std::int32_t M2CMD_CARD_STOP         = 0x00000002;  /*!< Stop card */
constexpr std::int32_t M2CMD_CARD_FORCESTOP    = 0x00000004;  /*!< Force stop card */
constexpr std::int32_t M2CMD_CARD_RESET        = 0x00000008;  /*!< Reset card */
constexpr std::int32_t M2CMD_DATA_STARTDMA     = 0x00000010;  /*!< Start DMA */
constexpr std::int32_t M2CMD_DATA_STOPDMA      = 0x00000020;  /*!< Stop DMA */

// Status bits
constexpr std::int32_t M2STAT_CARD_PRETRIGGER  = 0x00000001;  /*!< Pretrigger status */
constexpr std::int32_t M2STAT_CARD_TRIGGER     = 0x00000002;  /*!< Trigger status */
constexpr std::int32_t M2STAT_CARD_READY       = 0x00000004;  /*!< Ready status */
constexpr std::int32_t M2STAT_CARD_SEGMENT_PRESTART = 0x00000008; /*!< Segment prestart */
constexpr std::int32_t M2STAT_DATA_BLOCKREADY  = 0x00000100;  /*!< Data block ready */
constexpr std::int32_t M2STAT_DATA_END          = 0x00000200;  /*!< Data end */
constexpr std::int32_t M2STAT_DATA_OVERRUN     = 0x00000400;  /*!< Data overrun */

// Buffer Types for Transfer
constexpr std::int32_t SPCM_BUF_DATA           = 0x00000000;  /*!< Data buffer */
constexpr std::int32_t SPCM_BUF_ABA            = 0x00002000;  /*!< ABA buffer */
constexpr std::int32_t SPCM_BUF_TIMESTAMP      = 0x00004000;  /*!< Timestamp buffer */

// Transfer Directions  
constexpr std::int32_t SPCM_DIR_PCTOCARD       = 0x00000000;  /*!< PC to card */
constexpr std::int32_t SPCM_DIR_CARDTOPC       = 0x00000001;  /*!< Card to PC */

} // namespace Spectrum::M4i

#endif // SPECTRUMCONSTANTS_H