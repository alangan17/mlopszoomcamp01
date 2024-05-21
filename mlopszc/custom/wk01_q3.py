if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@custom
def transform_custom(*args, **kwargs):
    """
    args: The output from any upstream parent blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    df = args[0]
    df_row_count_before = len(df)
    print(f"Row count before dropping outliers: {df_row_count_before}")

    df = df[(df.duration >= 1) & (df.duration <= 60)]
    df_row_count_after = len(df)
    print(f"Row count after dropping outliers: {df_row_count_after}")

    print(f"fraction of the records left: {df_row_count_after/df_row_count_before*100}%")

    return df


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
