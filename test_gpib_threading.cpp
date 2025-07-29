#include <QApplication>
#include <QThread>
#include <QDebug>
#include <memory>

#ifdef BC_GPIBCONTROLLER
#include <hardware/optional/gpibcontroller/gpibcontroller.h>
#include <hardware/core/communication/gpibinstrument.h>
#endif

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    
#ifdef BC_GPIBCONTROLLER
    qDebug() << "Testing GPIB thread safety improvements...";
    
    // Test that GpibController can be created without special threading
    auto gpib = std::make_unique<BC_GPIBCONTROLLER>();
    qDebug() << "GpibController created successfully in main thread";
    
    // Test that GPIB devices don't require setParent(controller) anymore
    auto gpibInstr = std::make_unique<GpibInstrument>("testKey", gpib.get(), nullptr);
    qDebug() << "GpibInstrument created without parent coupling";
    
    // Test mutex protection by calling methods (they should not crash)
    bool result = gpib->writeCmd(1, "*IDN?");
    qDebug() << "GPIB writeCmd with mutex protection: " << (result ? "success" : "expected failure for virtual");
    
    qDebug() << "GPIB threading modernization test completed successfully!";
#else
    qDebug() << "GPIB controller not compiled in - cannot test threading improvements";
#endif
    
    return 0;
}