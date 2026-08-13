#include <QtTest/QtTest>

#include <QWidget>

#include <gui/widget/led.h>

/*!
 * \brief Proves that a test target can link the whole GUI library.
 *
 * AUTOMOC emits one mocs_compilation.cpp per target, so the moc for any
 * single Q_OBJECT GUI class drags in every other one — MainWindow
 * included, and with it the acquisition and hardware layers. Constructing
 * one trivial widget is enough to exercise that: if the link resolves and
 * the widget survives construction and destruction, the GUI stack is
 * reachable from a test.
 *
 * Failures here are structural rather than behavioural. Treat a break as
 * a report about the library graph or the blackchirp_add_gui_test() CMake
 * helper, not about Led.
 */
class TestGuiLinkage : public QObject
{
    Q_OBJECT

private slots:
    void testConstructAndDestroyWidget();
    void testParentedWidgetSurvivesParentDestruction();
};

void TestGuiLinkage::testConstructAndDestroyWidget()
{
    auto led = std::make_unique<Led>(Led::Green, 15);
    QVERIFY(led != nullptr);

    led->setState(true);
    led->setColor(Led::Red);
    led->setLedSize(24);

    led.reset();
}

void TestGuiLinkage::testParentedWidgetSurvivesParentDestruction()
{
    auto parent = std::make_unique<QWidget>();
    auto *led = new Led(Led::Blue, 15, parent.get());
    QCOMPARE(led->parentWidget(), parent.get());

    // Destroying the parent must take the child with it without the
    // child's destructor reaching back into a half-destroyed ancestor.
    parent.reset();
}

QTEST_MAIN(TestGuiLinkage)
#include "tst_guilinkage.moc"
