#include <unity.h>
#include <string.h>
#include "param_store_format.h"

void setUp() {}
void tearDown() {}

static ParamStoreHeader make_header(const ParamStoreEntry* entries, uint16_t count,
                                    uint32_t generation) {
    ParamStoreHeader header{
        PARAM_STORE_MAGIC, PARAM_STORE_VERSION, sizeof(ParamStoreHeader), generation,
        count, (uint16_t)(count * sizeof(ParamStoreEntry)),
        param_store_crc32(entries, count * sizeof(ParamStoreEntry)), 0
    };
    header.header_crc32 = param_store_header_crc(header);
    return header;
}

void test_header_and_payload_crc_detect_corruption() {
    ParamStoreEntry entries[] = {{5, 0.0f}, {6, 1.0f}, {0x010C, 1.0f}};
    ParamStoreHeader header = make_header(entries, 3, 42);
    TEST_ASSERT_TRUE(param_store_header_valid(header));
    TEST_ASSERT_EQUAL_UINT32(header.payload_crc32,
                             param_store_crc32(entries, sizeof(entries)));

    ParamStoreHeader bad_header = header;
    bad_header.generation ^= 0x100;
    TEST_ASSERT_FALSE(param_store_header_valid(bad_header));

    ParamStoreEntry bad_entries[3];
    memcpy(bad_entries, entries, sizeof(entries));
    bad_entries[1].value = 2.0f;
    TEST_ASSERT_NOT_EQUAL(header.payload_crc32,
                          param_store_crc32(bad_entries, sizeof(bad_entries)));
}

void test_structural_length_and_version_are_guarded() {
    ParamStoreEntry entry{5, 0.0f};
    ParamStoreHeader header = make_header(&entry, 1, 1);
    header.payload_bytes--;
    header.header_crc32 = param_store_header_crc(header);
    TEST_ASSERT_FALSE(param_store_header_valid(header));
    header = make_header(&entry, 1, 1);
    header.version++;
    header.header_crc32 = param_store_header_crc(header);
    TEST_ASSERT_FALSE(param_store_header_valid(header));
}

void test_generation_selection_handles_wrap() {
    TEST_ASSERT_TRUE(param_store_generation_newer(11, 10));
    TEST_ASSERT_FALSE(param_store_generation_newer(10, 11));
    TEST_ASSERT_TRUE(param_store_generation_newer(1, 0xFFFFFFFEu));
}

int main(int, char**) {
    UNITY_BEGIN();
    RUN_TEST(test_header_and_payload_crc_detect_corruption);
    RUN_TEST(test_structural_length_and_version_are_guarded);
    RUN_TEST(test_generation_selection_handles_wrap);
    return UNITY_END();
}
