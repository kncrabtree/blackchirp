#include <QtTest>
#include <QCoreApplication>
#include <QSettings>
#include <QThread>
#include <QSemaphore>
#include <QMutex>
#include <QTemporaryDir>
#include <QAtomicInt>

#include <src/hardware/core/hardwareprofilemanager.h>
#include <src/hardware/core/hardwareregistry.h>
#include <src/hardware/core/hardwareobject.h>

/*!
 * \brief Comprehensive unit tests for HardwareProfileManager
 * 
 * These tests define the API contract for the label-based hardware profile system
 * that will replace the current index-based hardware identification.
 * 
 * Test-driven development approach:
 * 1. Define expected behavior through tests
 * 2. Implement class to satisfy tests  
 * 3. Ensure thread safety and persistence
 */
class HardwareProfileManagerTest : public QObject
{
    Q_OBJECT

public:
    HardwareProfileManagerTest();
    ~HardwareProfileManagerTest();

private slots:
    void initTestCase();
    void cleanupTestCase();
    void init();
    void cleanup();
    
    // ========================================================================
    // CORE PROFILE MANAGEMENT TESTS
    // ========================================================================
    
    /*!
     * \brief Test basic profile creation with auto-generated labels
     */
    void testCreateProfile();
    
    /*!
     * \brief Test profile creation with custom user-provided labels
     */
    void testCreateProfileWithCustomLabel();
    
    /*!
     * \brief Test automatic label generation algorithms
     */
    void testCreateProfileAutoLabel();
    
    /*!
     * \brief Test profile deletion and cleanup
     */
    void testDeleteProfile();
    
    /*!
     * \brief Test profile activation and deactivation
     */
    void testActivateDeactivateProfile();
    
    /*!
     * \brief Test creating multiple profiles for same hardware type
     */
    void testMultipleProfilesSameType();
    
    // ========================================================================
    // LABEL MANAGEMENT TESTS
    // ========================================================================
    
    /*!
     * \brief Test label uniqueness enforcement within hardware types
     */
    void testLabelUniquenessSameType();
    
    /*!
     * \brief Test that labels can be reused across different hardware types
     */
    void testLabelReuseAcrossTypes();
    
    /*!
     * \brief Test default label generation patterns
     */
    void testDefaultLabelGeneration();
    
    /*!
     * \brief Test label validation rules and restrictions
     */
    void testLabelValidation();
    
    /*!
     * \brief Test label availability checking
     */
    void testLabelAvailability();
    
    /*!
     * \brief Test getting existing labels for a hardware type
     */
    void testGetExistingLabels();
    
    // ========================================================================
    // SETTINGS PERSISTENCE TESTS
    // ========================================================================
    
    /*!
     * \brief Test profile data persistence across application restarts
     */
    void testProfilePersistence();
    
    /*!
     * \brief Test correct storage format in QSettings
     */
    void testStorageFormat();
    
    /*!
     * \brief Test loading existing profiles from settings
     */
    void testLoadingExistingProfiles();
    
    /*!
     * \brief Test settings corruption recovery
     */
    void testSettingsCorruptionRecovery();
    
    /*!
     * \brief Test profile metadata (creation time, description, etc.)
     */
    void testProfileMetadata();
    
    // ========================================================================
    // COLLISION HANDLING TESTS
    // ========================================================================
    
    /*!
     * \brief Test collision detection when labels conflict
     */
    void testCollisionDetection();
    
    /*!
     * \brief Test collision resolution strategies
     */
    void testCollisionResolution();
    
    /*!
     * \brief Test collision handling during profile import
     */
    void testCollisionDuringImport();
    
    // ========================================================================
    // QUERY OPERATIONS TESTS
    // ========================================================================
    
    /*!
     * \brief Test getting active profiles for hardware type
     */
    void testGetActiveProfiles();
    
    /*!
     * \brief Test getting inactive profiles for hardware type
     */
    void testGetInactiveProfiles();
    
    /*!
     * \brief Test getting implementation for specific profile
     */
    void testGetImplementation();
    
    /*!
     * \brief Test getting all profiles for hardware type
     */
    void testGetAllProfiles();
    
    /*!
     * \brief Test profile existence checking
     */
    void testProfileExists();
    
    // ========================================================================
    // EDGE CASES AND ERROR HANDLING TESTS
    // ========================================================================
    
    /*!
     * \brief Test invalid inputs and error handling
     */
    void testInvalidInputs();
    
    /*!
     * \brief Test empty and null label handling
     */
    void testEmptyLabels();
    
    /*!
     * \brief Test maximum label length enforcement
     */
    void testMaxLabelLength();
    
    /*!
     * \brief Test special characters in labels
     */
    void testSpecialCharactersInLabels();
    
    /*!
     * \brief Test operations on non-existent profiles
     */
    void testNonExistentProfiles();
    
    /*!
     * \brief Test operations on non-existent hardware types
     */
    void testNonExistentHardwareTypes();
    
    // ========================================================================
    // THREAD SAFETY TESTS
    // ========================================================================
    
    /*!
     * \brief Test concurrent profile creation
     */
    void testConcurrentProfileCreation();
    
    /*!
     * \brief Test concurrent read operations
     */
    void testConcurrentReads();
    
    /*!
     * \brief Test mixed read/write operations under concurrency
     */
    void testConcurrentReadWrite();
    
    // ========================================================================
    // INTEGRATION TESTS
    // ========================================================================
    
    /*!
     * \brief Test integration with SettingsStorage
     */
    void testSettingsStorageIntegration();
    
    /*!
     * \brief Test profile import/export functionality
     */
    void testProfileImportExport();
    
    /*!
     * \brief Test bulk operations on multiple profiles
     */
    void testBulkOperations();

private:
    /*!
     * \brief Helper function to create test profiles
     */
    void createTestProfiles();
    
    /*!
     * \brief Helper function to verify profile state
     */
    void verifyProfileState(const QString& type, const QString& label, 
                           const QString& implementation, bool active);
    
    /*!
     * \brief Helper function to clear test data
     */
    void clearTestData();
    
    /*!
     * \brief Helper function to setup concurrent test environment
     */
    void setupConcurrentTest(int threadCount);
    
    // Test data
    QTemporaryDir* d_tempDir;
    QString d_testOrg = "CrabtreeLab";
    QString d_testApp = "BlackchirpHardwareProfileTest";
    
    // Common test hardware types and implementations
    QString d_testTypeFlow = "FlowController";
    QString d_testTypeDigitizer = "FtmwDigitizer";
    QString d_testTypeAWG = "ChirpSource";
    
    QString d_testImplVirtual = "virtual";
    QString d_testImplMks647c = "mks647c";
    QString d_testImplM4i = "m4i2220x8";
    QString d_testImplM8195 = "m8195a";
};

HardwareProfileManagerTest::HardwareProfileManagerTest()
{
    QCoreApplication::setApplicationName(d_testApp);
    QCoreApplication::setOrganizationName(d_testOrg);
    QCoreApplication::setOrganizationDomain("crabtreelab.ucdavis.edu");
}

HardwareProfileManagerTest::~HardwareProfileManagerTest()
{
}

void HardwareProfileManagerTest::initTestCase()
{
    // Create temporary directory for test settings
    d_tempDir = new QTemporaryDir();
    QVERIFY(d_tempDir->isValid());
    
    // Override QSettings location to use temporary directory
    QSettings::setPath(QSettings::IniFormat, QSettings::UserScope, d_tempDir->path());
    
    // Register test hardware implementations for validation
    HardwareRegistry& registry = HardwareRegistry::instance();
    registry.registerHardware(d_testTypeFlow, d_testImplVirtual, "Virtual Flow Controller for testing", 
                             [](const QString& label) -> HardwareObject* { Q_UNUSED(label) return nullptr; }); // Dummy factory for testing
    registry.registerHardware(d_testTypeFlow, d_testImplMks647c, "MKS 647C Flow Controller for testing", 
                             [](const QString& label) -> HardwareObject* { Q_UNUSED(label) return nullptr; }); // Dummy factory for testing
}

void HardwareProfileManagerTest::cleanupTestCase()
{
    delete d_tempDir;
}

void HardwareProfileManagerTest::init()
{
    // Clear any previous test data before each test
    clearTestData();
}

void HardwareProfileManagerTest::cleanup()
{
    // Clean up after each test
    clearTestData();
}

// ========================================================================
// CORE PROFILE MANAGEMENT TESTS
// ========================================================================

void HardwareProfileManagerTest::testCreateProfile()
{
    HardwareProfileManager manager(d_testOrg, d_testApp);
    
    // Test basic profile creation with auto-generated label
    QString label = manager.createHardwareProfile(d_testTypeFlow, d_testImplVirtual);
    
    // Verify label was generated and is not empty
    QVERIFY(!label.isEmpty());
    QVERIFY(manager.profileExists(d_testTypeFlow, label));
    
    // Verify profile is active by default
    QVERIFY(manager.isProfileActive(d_testTypeFlow, label));
    
    // Verify implementation is stored correctly
    QCOMPARE(manager.getImplementation(d_testTypeFlow, label), d_testImplVirtual);
    
    // Verify profile appears in active profiles list
    QStringList activeProfiles = manager.getActiveProfiles(d_testTypeFlow);
    QVERIFY(activeProfiles.contains(label));
}

void HardwareProfileManagerTest::testCreateProfileWithCustomLabel()
{
    HardwareProfileManager manager(d_testOrg, d_testApp);
    
    QString customLabel = "mainFlowController";
    QString label = manager.createHardwareProfile(d_testTypeFlow, d_testImplMks647c, customLabel);
    
    // Verify custom label was used
    QCOMPARE(label, customLabel);
    QVERIFY(manager.profileExists(d_testTypeFlow, label));
    QCOMPARE(manager.getImplementation(d_testTypeFlow, label), d_testImplMks647c);
}

void HardwareProfileManagerTest::testCreateProfileAutoLabel()
{
    HardwareProfileManager manager(d_testOrg, d_testApp);
    
    // Create multiple profiles with auto-generated labels
    QString label1 = manager.createHardwareProfile(d_testTypeFlow, d_testImplVirtual);
    QString label2 = manager.createHardwareProfile(d_testTypeFlow, d_testImplMks647c);
    QString label3 = manager.createHardwareProfile(d_testTypeFlow, d_testImplVirtual);
    
    // Verify all labels are unique
    QVERIFY(label1 != label2);
    QVERIFY(label1 != label3);
    QVERIFY(label2 != label3);
    
    // Verify all profiles exist
    QVERIFY(manager.profileExists(d_testTypeFlow, label1));
    QVERIFY(manager.profileExists(d_testTypeFlow, label2));
    QVERIFY(manager.profileExists(d_testTypeFlow, label3));
    
    // Test expected default label pattern (e.g., "Default", "Secondary", "Backup")
    QVERIFY(label1.contains("Default") || label1.contains("flow"));
    QVERIFY(!label2.isEmpty() && !label3.isEmpty());
}

void HardwareProfileManagerTest::testDeleteProfile()
{
    HardwareProfileManager manager(d_testOrg, d_testApp);
    
    // Create test profile
    QString label = manager.createHardwareProfile(d_testTypeFlow, d_testImplVirtual, "testDelete");
    QVERIFY(manager.profileExists(d_testTypeFlow, label));
    
    // Delete profile
    bool deleted = manager.deleteHardwareProfile(d_testTypeFlow, label);
    QVERIFY(deleted);
    QVERIFY(!manager.profileExists(d_testTypeFlow, label));
    
    // Verify profile no longer appears in any lists
    QVERIFY(!manager.getActiveProfiles(d_testTypeFlow).contains(label));
    QVERIFY(!manager.getInactiveProfiles(d_testTypeFlow).contains(label));
    
    // Test deleting non-existent profile
    bool deletedAgain = manager.deleteHardwareProfile(d_testTypeFlow, label);
    QVERIFY(!deletedAgain);
}

void HardwareProfileManagerTest::testActivateDeactivateProfile()
{
    HardwareProfileManager manager(d_testOrg, d_testApp);
    
    QString label = manager.createHardwareProfile(d_testTypeFlow, d_testImplVirtual, "testActivation");
    
    // Profile should be active by default
    QVERIFY(manager.isProfileActive(d_testTypeFlow, label));
    QVERIFY(manager.getActiveProfiles(d_testTypeFlow).contains(label));
    
    // Deactivate profile
    bool deactivated = manager.deactivateHardwareProfile(d_testTypeFlow, label);
    QVERIFY(deactivated);
    QVERIFY(!manager.isProfileActive(d_testTypeFlow, label));
    QVERIFY(manager.getInactiveProfiles(d_testTypeFlow).contains(label));
    
    // Reactivate profile
    bool activated = manager.activateHardwareProfile(d_testTypeFlow, label);
    QVERIFY(activated);
    QVERIFY(manager.isProfileActive(d_testTypeFlow, label));
    QVERIFY(manager.getActiveProfiles(d_testTypeFlow).contains(label));
}

void HardwareProfileManagerTest::testMultipleProfilesSameType()
{
    HardwareProfileManager manager(d_testOrg, d_testApp);
    
    // Create multiple profiles for same hardware type
    QString label1 = manager.createHardwareProfile(d_testTypeFlow, d_testImplVirtual, "primary");
    QString label2 = manager.createHardwareProfile(d_testTypeFlow, d_testImplMks647c, "secondary");
    QString label3 = manager.createHardwareProfile(d_testTypeFlow, d_testImplVirtual, "backup");
    
    // Verify all profiles exist and are independent
    QVERIFY(manager.profileExists(d_testTypeFlow, label1));
    QVERIFY(manager.profileExists(d_testTypeFlow, label2));
    QVERIFY(manager.profileExists(d_testTypeFlow, label3));
    
    // Verify different implementations
    QCOMPARE(manager.getImplementation(d_testTypeFlow, label1), d_testImplVirtual);
    QCOMPARE(manager.getImplementation(d_testTypeFlow, label2), d_testImplMks647c);
    QCOMPARE(manager.getImplementation(d_testTypeFlow, label3), d_testImplVirtual);
    
    // Test independent activation states
    manager.deactivateHardwareProfile(d_testTypeFlow, label2);
    QVERIFY(manager.isProfileActive(d_testTypeFlow, label1));
    QVERIFY(!manager.isProfileActive(d_testTypeFlow, label2));
    QVERIFY(manager.isProfileActive(d_testTypeFlow, label3));
}

// ========================================================================
// LABEL MANAGEMENT TESTS
// ========================================================================

void HardwareProfileManagerTest::testLabelUniquenessSameType()
{
    HardwareProfileManager manager(d_testOrg, d_testApp);
    
    QString label1 = manager.createHardwareProfile(d_testTypeFlow, d_testImplVirtual, "unique");
    QCOMPARE(label1, QString("unique"));
    
    // Try to create another profile with same label - should fail or auto-rename
    QString label2 = manager.createHardwareProfile(d_testTypeFlow, d_testImplMks647c, "unique");
    
    // Should either fail (empty return) or auto-generate new label
    if (!label2.isEmpty()) {
        QVERIFY(label2 != label1); // Auto-renamed
        QVERIFY(manager.profileExists(d_testTypeFlow, label2));
    }
    
    // Test label availability checking
    QVERIFY(!manager.isLabelAvailable(d_testTypeFlow, "unique"));
    QVERIFY(manager.isLabelAvailable(d_testTypeFlow, "availableLabel"));
}

void HardwareProfileManagerTest::testLabelReuseAcrossTypes()
{
    HardwareProfileManager manager(d_testOrg, d_testApp);
    
    QString sameLabel = "mainDevice";
    
    // Create profiles with same label for different hardware types
    QString label1 = manager.createHardwareProfile(d_testTypeFlow, d_testImplVirtual, sameLabel);
    QString label2 = manager.createHardwareProfile(d_testTypeDigitizer, d_testImplM4i, sameLabel);
    
    // Both should succeed with same label
    QCOMPARE(label1, sameLabel);
    QCOMPARE(label2, sameLabel);
    
    // Verify both profiles exist independently
    QVERIFY(manager.profileExists(d_testTypeFlow, sameLabel));
    QVERIFY(manager.profileExists(d_testTypeDigitizer, sameLabel));
    
    // Verify different implementations
    QCOMPARE(manager.getImplementation(d_testTypeFlow, sameLabel), d_testImplVirtual);
    QCOMPARE(manager.getImplementation(d_testTypeDigitizer, sameLabel), d_testImplM4i);
}

void HardwareProfileManagerTest::testDefaultLabelGeneration()
{
    HardwareProfileManager manager(d_testOrg, d_testApp);
    
    // Test default label generation for empty existing labels
    QString defaultLabel1 = manager.generateDefaultLabel(d_testTypeFlow);
    QVERIFY(!defaultLabel1.isEmpty());
    QVERIFY(manager.isLabelAvailable(d_testTypeFlow, defaultLabel1));
    
    // Create profile with default label
    manager.createHardwareProfile(d_testTypeFlow, d_testImplVirtual, defaultLabel1);
    
    // Generate second default label (should be different)
    QString defaultLabel2 = manager.generateDefaultLabel(d_testTypeFlow);
    QVERIFY(!defaultLabel2.isEmpty());
    QVERIFY(defaultLabel2 != defaultLabel1);
    QVERIFY(manager.isLabelAvailable(d_testTypeFlow, defaultLabel2));
    
    // Test pattern (e.g., "Default", "Secondary", "Backup", "flow1", "flow2", etc.)
    QVERIFY(defaultLabel1.contains("Default") ||
            defaultLabel1.toLower().contains(d_testTypeFlow.toLower()) ||
            defaultLabel1.contains("Main"));
}

void HardwareProfileManagerTest::testLabelValidation()
{
    HardwareProfileManager manager(d_testOrg, d_testApp);
    
    // Test valid labels
    QVERIFY(manager.isValidLabel("validLabel"));
    QVERIFY(manager.isValidLabel("main_device"));
    QVERIFY(manager.isValidLabel("device-01"));
    QVERIFY(manager.isValidLabel("Device123"));
    
    // Test invalid labels
    QVERIFY(!manager.isValidLabel(""));           // Empty
    QVERIFY(!manager.isValidLabel("   "));        // Whitespace only
    QVERIFY(!manager.isValidLabel("label with spaces")); // Internal spaces
    QVERIFY(!manager.isValidLabel("label.with.dots"));   // Dots (conflicts with key format)
    QVERIFY(!manager.isValidLabel("very_long_label_name_that_definitely_exceeds_reasonable_length_limits_for_hardware_profiles")); // Too long
    
    // Test edge cases
    QVERIFY(!manager.isValidLabel("123_starts_with_number")); // Starts with number
    QVERIFY(!manager.isValidLabel("_starts_with_underscore")); // Starts with underscore
    QVERIFY(manager.isValidLabel("a"));           // Single character
    QVERIFY(manager.isValidLabel("A1"));          // Letter + number
}

void HardwareProfileManagerTest::testLabelAvailability()
{
    HardwareProfileManager manager(d_testOrg, d_testApp);
    
    QString testLabel = "availabilityTest";
    
    // Label should be available initially
    QVERIFY(manager.isLabelAvailable(d_testTypeFlow, testLabel));
    
    // Create profile with label
    manager.createHardwareProfile(d_testTypeFlow, d_testImplVirtual, testLabel);
    
    // Label should no longer be available for same type
    QVERIFY(!manager.isLabelAvailable(d_testTypeFlow, testLabel));
    
    // But should still be available for different type
    QVERIFY(manager.isLabelAvailable(d_testTypeDigitizer, testLabel));
}

void HardwareProfileManagerTest::testGetExistingLabels()
{
    HardwareProfileManager manager(d_testOrg, d_testApp);
    
    // Initially no labels
    QVERIFY(manager.getExistingLabels(d_testTypeFlow).isEmpty());
    
    // Create several profiles
    manager.createHardwareProfile(d_testTypeFlow, d_testImplVirtual, "label1");
    manager.createHardwareProfile(d_testTypeFlow, d_testImplMks647c, "label2");
    manager.createHardwareProfile(d_testTypeDigitizer, d_testImplM4i, "label3"); // Different type
    
    QStringList flowLabels = manager.getExistingLabels(d_testTypeFlow);
    QCOMPARE(flowLabels.size(), 2);
    QVERIFY(flowLabels.contains("label1"));
    QVERIFY(flowLabels.contains("label2"));
    QVERIFY(!flowLabels.contains("label3")); // Different type
    
    QStringList digitizerLabels = manager.getExistingLabels(d_testTypeDigitizer);
    QCOMPARE(digitizerLabels.size(), 1);
    QVERIFY(digitizerLabels.contains("label3"));
}

// ========================================================================
// SETTINGS PERSISTENCE TESTS
// ========================================================================

void HardwareProfileManagerTest::testProfilePersistence()
{
    // Create profiles in first manager instance
    {
        HardwareProfileManager manager(d_testOrg, d_testApp);
        manager.createHardwareProfile(d_testTypeFlow, d_testImplVirtual, "persistent1");
        manager.createHardwareProfile(d_testTypeFlow, d_testImplMks647c, "persistent2");
        manager.deactivateHardwareProfile(d_testTypeFlow, "persistent2");
    } // Manager destroyed, should save to settings
    
    // Create new manager instance and verify profiles are loaded
    {
        HardwareProfileManager manager2(d_testOrg, d_testApp);
        
        QVERIFY(manager2.profileExists(d_testTypeFlow, "persistent1"));
        QVERIFY(manager2.profileExists(d_testTypeFlow, "persistent2"));
        
        QCOMPARE(manager2.getImplementation(d_testTypeFlow, "persistent1"), d_testImplVirtual);
        QCOMPARE(manager2.getImplementation(d_testTypeFlow, "persistent2"), d_testImplMks647c);
        
        QVERIFY(manager2.isProfileActive(d_testTypeFlow, "persistent1"));
        QVERIFY(!manager2.isProfileActive(d_testTypeFlow, "persistent2"));
    }
}

void HardwareProfileManagerTest::testStorageFormat()
{
    HardwareProfileManager manager(d_testOrg, d_testApp);
    
    manager.createHardwareProfile(d_testTypeFlow, d_testImplVirtual, "testFormat");
    manager.saveProfiles(); // Force save
    
    // Check storage format using SettingsStorage read-only access
    SettingsStorage storage(d_testOrg, d_testApp, {"HardwareProfiles"});
    
    // Expected format: Group "FlowController.testFormat" with keys "implementation", "active", "created"
    QString profileGroup = QString("%1.%2").arg(d_testTypeFlow, "testFormat");
    
    // Check that the group exists in the stored data
    QStringList groupKeys = storage.groupKeys();
    QVERIFY(groupKeys.contains(profileGroup));
    
    // Check the values within the group using getGroupValue
    QCOMPARE(storage.getGroupValue<QString>(profileGroup, "implementation"), d_testImplVirtual);
    QCOMPARE(storage.getGroupValue<bool>(profileGroup, "active"), true);
    QVERIFY(!storage.getGroupValue<QString>(profileGroup, "created").isEmpty());
}

void HardwareProfileManagerTest::testLoadingExistingProfiles()
{
    // Manually create settings entries using SettingsStorage interface
    {
        HardwareProfileManager manager(d_testOrg, d_testApp);
        
        // Use SettingsStorage GroupValues to create "preexisting" profile  
        QString groupKey = "FlowController.preexisting";
        manager.setGroupValue(groupKey, "implementation", QString("mks647c"), false);
        manager.setGroupValue(groupKey, "active", false, false);
        manager.setGroupValue(groupKey, "created", QString("2024-01-15T10:30:00"), false);
        manager.setGroupValue(groupKey, "description", QString("Pre-existing test profile"), false);
        
        // Save the settings
        manager.save();
    } // Manager destroyed
    
    // Create new manager and verify it loads existing profiles
    HardwareProfileManager manager(d_testOrg, d_testApp);
    
    QVERIFY(manager.profileExists(d_testTypeFlow, "preexisting"));
    QCOMPARE(manager.getImplementation(d_testTypeFlow, "preexisting"), QString("mks647c"));
    QVERIFY(!manager.isProfileActive(d_testTypeFlow, "preexisting"));
}

void HardwareProfileManagerTest::testSettingsCorruptionRecovery()
{
    // Create corrupted settings entries using SettingsStorage interface
    {
        HardwareProfileManager manager(d_testOrg, d_testApp);
        
        // Incomplete profile (missing implementation)
        QString corruptedGroup1 = "FlowController.corrupted1";
        manager.setGroupValue(corruptedGroup1, "active", true, false);
        manager.setGroupValue(corruptedGroup1, "created", QString("invalid-date"), false);
        // Note: intentionally missing "implementation" key
        
        // Invalid data types
        QString corruptedGroup2 = "FlowController.corrupted2";
        manager.setGroupValue(corruptedGroup2, "implementation", 12345, false); // Number instead of string
        manager.setGroupValue(corruptedGroup2, "active", QString("not-a-bool"), false); // String instead of bool
        
        manager.save();
    } // Manager destroyed, settings persisted
    
    // Manager should handle corruption gracefully
    HardwareProfileManager manager(d_testOrg, d_testApp);
    
    // Corrupted profiles should be ignored or cleaned up
    QVERIFY(!manager.profileExists(d_testTypeFlow, "corrupted1") || 
            manager.getImplementation(d_testTypeFlow, "corrupted1").isEmpty());
    QVERIFY(!manager.profileExists(d_testTypeFlow, "corrupted2") || 
            manager.getImplementation(d_testTypeFlow, "corrupted2").isEmpty());
}

void HardwareProfileManagerTest::testProfileMetadata()
{
    HardwareProfileManager manager(d_testOrg, d_testApp);
    
    QString label = manager.createHardwareProfile(d_testTypeFlow, d_testImplVirtual, "metadataTest");
    
    // Check creation timestamp
    QDateTime created = manager.getProfileCreationTime(d_testTypeFlow, label);
    QVERIFY(created.isValid());
    QVERIFY(created <= QDateTime::currentDateTime());
    QVERIFY(created >= QDateTime::currentDateTime().addSecs(-10)); // Created within last 10 seconds
    
    // Test description setting/getting
    QString description = "Test flow controller for metadata testing";
    manager.setProfileDescription(d_testTypeFlow, label, description);
    QCOMPARE(manager.getProfileDescription(d_testTypeFlow, label), description);
    
    // Test last modified timestamp
    QDateTime modified = manager.getProfileLastModified(d_testTypeFlow, label);
    QVERIFY(modified.isValid());
    QVERIFY(modified >= created);
}

// ========================================================================
// COLLISION HANDLING TESTS
// ========================================================================

void HardwareProfileManagerTest::testCollisionDetection()
{
    HardwareProfileManager manager(d_testOrg, d_testApp);
    
    // Create initial profile
    manager.createHardwareProfile(d_testTypeFlow, d_testImplVirtual, "collision");
    
    // Detect collision on second creation
    HardwareProfileManager::CollisionAction action = 
        manager.detectCollision(d_testTypeFlow, "collision", d_testImplMks647c);
    
    QVERIFY(action != HardwareProfileManager::NoCollision);
}

void HardwareProfileManagerTest::testCollisionResolution()
{
    HardwareProfileManager manager(d_testOrg, d_testApp);
    
    manager.createHardwareProfile(d_testTypeFlow, d_testImplVirtual, "resolve");
    
    // Test different resolution strategies
    
    // 1. Rename strategy - should create with modified label
    QString renamedLabel = manager.createHardwareProfile(d_testTypeFlow, d_testImplMks647c, 
                                                        "resolve", HardwareProfileManager::Rename);
    QVERIFY(!renamedLabel.isEmpty());
    QVERIFY(renamedLabel != "resolve");
    QVERIFY(renamedLabel.startsWith("resolve"));
    
    // 2. Replace strategy - should replace existing profile
    QString replacedLabel = manager.createHardwareProfile(d_testTypeFlow, d_testImplMks647c,
                                                         "resolve", HardwareProfileManager::Replace);
    QCOMPARE(replacedLabel, QString("resolve"));
    QCOMPARE(manager.getImplementation(d_testTypeFlow, "resolve"), d_testImplMks647c);
}

void HardwareProfileManagerTest::testCollisionDuringImport()
{
    HardwareProfileManager manager(d_testOrg, d_testApp);
    
    // Create existing profile
    manager.createHardwareProfile(d_testTypeFlow, d_testImplVirtual, "import");
    
    // Simulate import of conflicting profile
    HardwareProfileData importData;
    importData.type = d_testTypeFlow;
    importData.label = "import";
    importData.implementation = d_testImplMks647c;
    importData.active = false;
    
    // Import with different collision strategies
    bool imported = manager.importProfile(importData, HardwareProfileManager::Rename);
    QVERIFY(imported);
    
    // Should have both profiles now
    QStringList labels = manager.getExistingLabels(d_testTypeFlow);
    QVERIFY(labels.size() >= 2);
    QVERIFY(labels.contains("import"));
}

// ========================================================================
// QUERY OPERATIONS TESTS
// ========================================================================

void HardwareProfileManagerTest::testGetActiveProfiles()
{
    HardwareProfileManager manager(d_testOrg, d_testApp);
    
    // Create mix of active and inactive profiles
    manager.createHardwareProfile(d_testTypeFlow, d_testImplVirtual, "active1");
    manager.createHardwareProfile(d_testTypeFlow, d_testImplMks647c, "active2");
    QString inactive = manager.createHardwareProfile(d_testTypeFlow, d_testImplVirtual, "inactive");
    manager.deactivateHardwareProfile(d_testTypeFlow, inactive);
    
    QStringList activeProfiles = manager.getActiveProfiles(d_testTypeFlow);
    
    QCOMPARE(activeProfiles.size(), 2);
    QVERIFY(activeProfiles.contains("active1"));
    QVERIFY(activeProfiles.contains("active2"));
    QVERIFY(!activeProfiles.contains("inactive"));
}

void HardwareProfileManagerTest::testGetInactiveProfiles()
{
    HardwareProfileManager manager(d_testOrg, d_testApp);
    
    // Create mix of active and inactive profiles
    manager.createHardwareProfile(d_testTypeFlow, d_testImplVirtual, "active");
    QString inactive1 = manager.createHardwareProfile(d_testTypeFlow, d_testImplMks647c, "inactive1");
    QString inactive2 = manager.createHardwareProfile(d_testTypeFlow, d_testImplVirtual, "inactive2");
    
    manager.deactivateHardwareProfile(d_testTypeFlow, inactive1);
    manager.deactivateHardwareProfile(d_testTypeFlow, inactive2);
    
    QStringList inactiveProfiles = manager.getInactiveProfiles(d_testTypeFlow);
    
    QCOMPARE(inactiveProfiles.size(), 2);
    QVERIFY(inactiveProfiles.contains("inactive1"));
    QVERIFY(inactiveProfiles.contains("inactive2"));
    QVERIFY(!inactiveProfiles.contains("active"));
}

void HardwareProfileManagerTest::testGetImplementation()
{
    HardwareProfileManager manager(d_testOrg, d_testApp);
    
    manager.createHardwareProfile(d_testTypeFlow, d_testImplVirtual, "getImpl");
    
    QCOMPARE(manager.getImplementation(d_testTypeFlow, "getImpl"), d_testImplVirtual);
    QVERIFY(manager.getImplementation(d_testTypeFlow, "nonexistent").isEmpty());
    QVERIFY(manager.getImplementation("NonexistentType", "getImpl").isEmpty());
}

void HardwareProfileManagerTest::testGetAllProfiles()
{
    HardwareProfileManager manager(d_testOrg, d_testApp);
    
    // Create profiles in different types
    manager.createHardwareProfile(d_testTypeFlow, d_testImplVirtual, "all1");
    manager.createHardwareProfile(d_testTypeFlow, d_testImplMks647c, "all2");
    manager.createHardwareProfile(d_testTypeDigitizer, d_testImplM4i, "all3");
    
    QStringList allFlow = manager.getAllProfiles(d_testTypeFlow);
    QStringList allDigitizer = manager.getAllProfiles(d_testTypeDigitizer);
    
    QCOMPARE(allFlow.size(), 2);
    QVERIFY(allFlow.contains("all1"));
    QVERIFY(allFlow.contains("all2"));
    
    QCOMPARE(allDigitizer.size(), 1);
    QVERIFY(allDigitizer.contains("all3"));
}

void HardwareProfileManagerTest::testProfileExists()
{
    HardwareProfileManager manager(d_testOrg, d_testApp);
    
    manager.createHardwareProfile(d_testTypeFlow, d_testImplVirtual, "exists");
    
    QVERIFY(manager.profileExists(d_testTypeFlow, "exists"));
    QVERIFY(!manager.profileExists(d_testTypeFlow, "doesnotexist"));
    QVERIFY(!manager.profileExists("NonexistentType", "exists"));
}

// ========================================================================
// EDGE CASES AND ERROR HANDLING TESTS
// ========================================================================

void HardwareProfileManagerTest::testInvalidInputs()
{
    HardwareProfileManager manager(d_testOrg, d_testApp);
    
    // Test empty/null inputs
    QString result1 = manager.createHardwareProfile("", d_testImplVirtual);
    QVERIFY(result1.isEmpty());
    
    QString result2 = manager.createHardwareProfile(d_testTypeFlow, "");
    QVERIFY(result2.isEmpty());
    
    // Test invalid hardware types
    QString result3 = manager.createHardwareProfile("InvalidHardwareType", d_testImplVirtual);
    // Should either fail or create anyway depending on validation level
    
    // Test operations on invalid profiles
    QVERIFY(!manager.deleteHardwareProfile("", ""));
    QVERIFY(!manager.activateHardwareProfile(d_testTypeFlow, ""));
    QVERIFY(manager.getImplementation("", "").isEmpty());
}

void HardwareProfileManagerTest::testEmptyLabels()
{
    HardwareProfileManager manager(d_testOrg, d_testApp);
    
    // Test creation with empty label (should auto-generate)
    QString label = manager.createHardwareProfile(d_testTypeFlow, d_testImplVirtual, "");
    QVERIFY(!label.isEmpty());
    
    // Test operations with empty labels
    QVERIFY(!manager.profileExists(d_testTypeFlow, ""));
    QVERIFY(!manager.isLabelAvailable(d_testTypeFlow, ""));
    QVERIFY(manager.getImplementation(d_testTypeFlow, "").isEmpty());
}

void HardwareProfileManagerTest::testMaxLabelLength()
{
    HardwareProfileManager manager(d_testOrg, d_testApp);
    
    // Test reasonable length label
    QString reasonableLabel = "reasonable_length_label_name";
    QString result1 = manager.createHardwareProfile(d_testTypeFlow, d_testImplVirtual, reasonableLabel);
    QCOMPARE(result1, reasonableLabel);
    
    // Test very long label
    QString longLabel = QString("very_long_label_name_").repeated(10); // 200+ characters
    QString result2 = manager.createHardwareProfile(d_testTypeFlow, d_testImplVirtual, longLabel);
    
    // Should either truncate, reject, or auto-generate alternative
    if (!result2.isEmpty()) {
        QVERIFY(result2.length() <= manager.getMaxLabelLength());
    }
}

void HardwareProfileManagerTest::testSpecialCharactersInLabels()
{
    HardwareProfileManager manager(d_testOrg, d_testApp);
    
    // Test various special characters
    QStringList testLabels = {
        "label_with_underscores",
        "label-with-dashes", 
        "label123with456numbers",
        "labelWithCamelCase",
        "LABEL_WITH_CAPS"
    };
    
    for (const QString& testLabel : testLabels) {
        QString result = manager.createHardwareProfile(d_testTypeFlow, d_testImplVirtual, testLabel);
        if (manager.isValidLabel(testLabel)) {
            QCOMPARE(result, testLabel);
        } else {
            // Should either reject or sanitize
            QVERIFY(result.isEmpty() || result != testLabel);
        }
    }
}

void HardwareProfileManagerTest::testNonExistentProfiles()
{
    HardwareProfileManager manager(d_testOrg, d_testApp);
    
    QString nonExistent = "doesNotExist";
    
    // All operations on non-existent profiles should fail gracefully
    QVERIFY(!manager.deleteHardwareProfile(d_testTypeFlow, nonExistent));
    QVERIFY(!manager.activateHardwareProfile(d_testTypeFlow, nonExistent));
    QVERIFY(!manager.deactivateHardwareProfile(d_testTypeFlow, nonExistent));
    QVERIFY(!manager.isProfileActive(d_testTypeFlow, nonExistent));
    QVERIFY(manager.getImplementation(d_testTypeFlow, nonExistent).isEmpty());
}

void HardwareProfileManagerTest::testNonExistentHardwareTypes()
{
    HardwareProfileManager manager(d_testOrg, d_testApp);
    
    QString nonExistentType = "NonExistentHardwareType";
    
    // Operations on non-existent types should fail gracefully
    QVERIFY(manager.getActiveProfiles(nonExistentType).isEmpty());
    QVERIFY(manager.getInactiveProfiles(nonExistentType).isEmpty());
    QVERIFY(manager.getExistingLabels(nonExistentType).isEmpty());
}

// ========================================================================
// THREAD SAFETY TESTS
// ========================================================================

void HardwareProfileManagerTest::testConcurrentProfileCreation()
{
    HardwareProfileManager manager(d_testOrg, d_testApp);
    
    const int threadCount = 10;
    const int profilesPerThread = 5;
    QVector<QThread*> threads;
    QVector<QString> results;
    QMutex resultsMutex;
    
    // Create threads that simultaneously create profiles
    for (int i = 0; i < threadCount; ++i) {
        QThread* thread = QThread::create([&manager, &results, &resultsMutex, i, profilesPerThread]() {
            for (int j = 0; j < profilesPerThread; ++j) {
                QString label = QString("thread%1_profile%2").arg(i).arg(j);
                QString result = manager.createHardwareProfile("ConcurrentTest", "virtual", label);
                
                QMutexLocker locker(&resultsMutex);
                results.append(result);
            }
        });
        threads.append(thread);
    }
    
    // Start all threads
    for (QThread* thread : threads) {
        thread->start();
    }
    
    // Wait for completion
    for (QThread* thread : threads) {
        QVERIFY(thread->wait(5000)); // 5 second timeout
        delete thread;
    }
    
    // Verify all profiles were created successfully
    QCOMPARE(results.size(), threadCount * profilesPerThread);
    
    // Verify no duplicate labels
    QSet<QString> uniqueResults = QSet<QString>(results.begin(), results.end());
    QCOMPARE(uniqueResults.size(), results.size());
}

void HardwareProfileManagerTest::testConcurrentReads()
{
    HardwareProfileManager manager(d_testOrg, d_testApp);
    
    // Create test data
    for (int i = 0; i < 10; ++i) {
        manager.createHardwareProfile(d_testTypeFlow, d_testImplVirtual, QString("read%1").arg(i));
    }
    
    const int threadCount = 20;
    QVector<QThread*> threads;
    QAtomicInt successCount;
    
    // Create threads that simultaneously read profiles
    for (int i = 0; i < threadCount; ++i) {
        QThread* thread = QThread::create([&manager, &successCount]() {
            // Perform various read operations
            QStringList active = manager.getActiveProfiles("FlowController");
            QStringList existing = manager.getExistingLabels("FlowController");
            bool exists = manager.profileExists("FlowController", "read0");
            QString impl = manager.getImplementation("FlowController", "read1");
            
            if (!active.isEmpty() && !existing.isEmpty() && exists && !impl.isEmpty()) {
                successCount.fetchAndAddOrdered(1);
            }
        });
        threads.append(thread);
    }
    
    // Start all threads
    for (QThread* thread : threads) {
        thread->start();
    }
    
    // Wait for completion
    for (QThread* thread : threads) {
        QVERIFY(thread->wait(5000));
        delete thread;
    }
    
    // All reads should have succeeded
    QCOMPARE(successCount.loadAcquire(), threadCount);
}

void HardwareProfileManagerTest::testConcurrentReadWrite()
{
    HardwareProfileManager manager(d_testOrg, d_testApp);
    
    const int readerCount = 10;
    const int writerCount = 5;
    QVector<QThread*> threads;
    QAtomicInt readerSuccess;
    QAtomicInt writerSuccess;
    
    // Create reader threads
    for (int i = 0; i < readerCount; ++i) {
        QThread* thread = QThread::create([&manager, &readerSuccess, i]() {
            for (int j = 0; j < 100; ++j) { // Many read operations
                QStringList profiles = manager.getActiveProfiles(QString("ReaderType%1").arg(i % 3));
                if (j % 10 == 0) { // Occasional check for specific profile
                    bool exists = manager.profileExists("WriterType", "writer0");
                    Q_UNUSED(exists)
                }
            }
            readerSuccess.fetchAndAddOrdered(1);
        });
        threads.append(thread);
    }
    
    // Create writer threads
    for (int i = 0; i < writerCount; ++i) {
        QThread* thread = QThread::create([&manager, &writerSuccess, i]() {
            QString label = QString("writer%1").arg(i);
            QString result = manager.createHardwareProfile("WriterType", "virtual", label);
            if (!result.isEmpty()) {
                writerSuccess.fetchAndAddOrdered(1);
            }
        });
        threads.append(thread);
    }
    
    // Start all threads
    for (QThread* thread : threads) {
        thread->start();
    }
    
    // Wait for completion
    for (QThread* thread : threads) {
        QVERIFY(thread->wait(10000)); // 10 second timeout
        delete thread;
    }
    
    // Verify operations completed successfully
    QCOMPARE(readerSuccess.loadAcquire(), readerCount);
    QCOMPARE(writerSuccess.loadAcquire(), writerCount);
}

// ========================================================================
// INTEGRATION TESTS
// ========================================================================

void HardwareProfileManagerTest::testSettingsStorageIntegration()
{
    // Test that HardwareProfileManager properly inherits SettingsStorage behavior
    HardwareProfileManager manager(d_testOrg, d_testApp);
    
    // Test SettingsStorage functionality through inheritance
    manager.set("testKey", QString("testValue"), true); // Use set() instead of setDefault() and write immediately
    QCOMPARE(manager.get<QString>("testKey"), QString("testValue"));
    
    // Test that profiles don't interfere with regular settings
    manager.createHardwareProfile(d_testTypeFlow, d_testImplVirtual, "integration");
    QCOMPARE(manager.get<QString>("testKey"), QString("testValue"));
    
    // Test settings persistence with profiles
    manager.saveProfiles();
    
    // Verify both regular settings and profiles are persisted
    HardwareProfileManager manager2(d_testOrg, d_testApp);
    QCOMPARE(manager2.get<QString>("testKey"), QString("testValue"));
    QVERIFY(manager2.profileExists(d_testTypeFlow, "integration"));
}

void HardwareProfileManagerTest::testProfileImportExport()
{
    HardwareProfileManager manager(d_testOrg, d_testApp);
    
    // Create test profiles
    manager.createHardwareProfile(d_testTypeFlow, d_testImplVirtual, "export1");
    manager.createHardwareProfile(d_testTypeFlow, d_testImplMks647c, "export2");
    manager.createHardwareProfile(d_testTypeDigitizer, d_testImplM4i, "export3");
    
    // Export profiles
    QByteArray exportData = manager.exportProfiles();
    QVERIFY(!exportData.isEmpty());
    
    // Clear current profiles
    manager.clearAllProfiles();
    QVERIFY(!manager.profileExists(d_testTypeFlow, "export1"));
    
    // Import profiles
    bool imported = manager.importProfiles(exportData);
    QVERIFY(imported);
    
    // Verify profiles were restored
    QVERIFY(manager.profileExists(d_testTypeFlow, "export1"));
    QVERIFY(manager.profileExists(d_testTypeFlow, "export2"));
    QVERIFY(manager.profileExists(d_testTypeDigitizer, "export3"));
    
    QCOMPARE(manager.getImplementation(d_testTypeFlow, "export1"), d_testImplVirtual);
    QCOMPARE(manager.getImplementation(d_testTypeFlow, "export2"), d_testImplMks647c);
    QCOMPARE(manager.getImplementation(d_testTypeDigitizer, "export3"), d_testImplM4i);
}

void HardwareProfileManagerTest::testBulkOperations()
{
    HardwareProfileManager manager(d_testOrg, d_testApp);
    
    // Create multiple profiles
    QStringList labels = {"bulk1", "bulk2", "bulk3", "bulk4", "bulk5"};
    for (const QString& label : labels) {
        manager.createHardwareProfile(d_testTypeFlow, d_testImplVirtual, label);
    }
    
    // Test bulk activation/deactivation
    bool deactivated = manager.deactivateAllProfiles(d_testTypeFlow);
    QVERIFY(deactivated);
    
    for (const QString& label : labels) {
        QVERIFY(!manager.isProfileActive(d_testTypeFlow, label));
    }
    
    // Test bulk reactivation
    bool activated = manager.activateAllProfiles(d_testTypeFlow);
    QVERIFY(activated);
    
    for (const QString& label : labels) {
        QVERIFY(manager.isProfileActive(d_testTypeFlow, label));
    }
    
    // Test bulk deletion
    bool deleted = manager.deleteAllProfiles(d_testTypeFlow);
    QVERIFY(deleted);
    
    for (const QString& label : labels) {
        QVERIFY(!manager.profileExists(d_testTypeFlow, label));
    }
}

// ========================================================================
// HELPER FUNCTIONS
// ========================================================================

void HardwareProfileManagerTest::createTestProfiles()
{
    // Helper to create standard test profiles for other tests
    // Implementation will be added as needed
}

void HardwareProfileManagerTest::verifyProfileState(const QString& type, const QString& label, 
                                                   const QString& implementation, bool active)
{
    HardwareProfileManager manager(d_testOrg, d_testApp);
    
    QVERIFY(manager.profileExists(type, label));
    QCOMPARE(manager.getImplementation(type, label), implementation);
    QCOMPARE(manager.isProfileActive(type, label), active);
}

void HardwareProfileManagerTest::clearTestData()
{
    // Clear all test data from settings
    QSettings settings(d_testOrg, d_testApp);
    settings.clear();
}

void HardwareProfileManagerTest::setupConcurrentTest(int threadCount)
{
    Q_UNUSED(threadCount)
    // Setup for concurrent testing
    clearTestData();
}

QTEST_MAIN(HardwareProfileManagerTest)
#include "tst_hardwareprofilemanagertest.moc"