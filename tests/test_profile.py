from tiny_chat.profiles import BaseAgentProfile, BaseEnvironmentProfile


def test_agent_profile():
    profile = BaseAgentProfile(
        first_name="John",
        last_name="Doe",
        age=25,
        occupation="Software Engineer",
    )

    profile.add_field("mbti", "intj")
    profile.remove_field("occupation")
    xml_profile = profile.to_background_string(agent_id=1)
    print(xml_profile)


def test_environment_profile():
    profile = BaseEnvironmentProfile(scenario="Three agents are eating dinner")
    print(profile)


if __name__ == "__main__":
    test_environment_profile()
